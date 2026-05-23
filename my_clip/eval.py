"""
零样本分类评估脚本

支持 CIFAR-10 和 CIFAR-100 的零样本分类评估，输出 Top-1 / Top-5 准确率。

实现方式:
    - 为每个类别构建 prompt 文本 (如 "a photo of a cat.")
    - 编码所有类别的文本嵌入 (多个模板取平均)
    - 编码所有测试图像
    - 计算图像与类别文本嵌入的 cosine 相似度, 取最高分为预测

用法:
    python -m my_clip.eval --checkpoint experiments/baseline/checkpoint/best_model.pt
    python -m my_clip.eval --checkpoint path/to/model.pt --datasets cifar10 cifar100
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from my_clip.config import CLIPConfig, EvalConfig
from my_clip.model import build_clip, CLIP


# ============================================================
# CIFAR 类别名称
# ============================================================

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR100_CLASSES = [
    "apple", "aquarium fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak tree", "orange", "orchid",
    "otter", "palm tree", "pear", "pickup truck", "pine tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow tree", "wolf", "woman", "worm",
]


def get_dataset_info(name: str) -> Tuple[List[str], callable]:
    """获取数据集对应的类别名称和 torchvision 数据集类。"""
    name = name.lower()
    if name == "cifar10":
        return CIFAR10_CLASSES, datasets.CIFAR10
    elif name == "cifar100":
        return CIFAR100_CLASSES, datasets.CIFAR100
    else:
        raise ValueError(f"不支持的数据集: {name}, 目前支持: cifar10, cifar100")


# ============================================================
# 零样本分类器
# ============================================================

@torch.no_grad()
def build_zero_shot_classifier(
    model: CLIP,
    class_names: List[str],
    prompt_templates: Tuple[str, ...],
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    构建零样本分类器权重矩阵。

    对每个类别, 用所有 prompt 模板填充类别名, 取文本嵌入的均值作为该类的代表向量。

    Args:
        model: CLIP 模型 (只使用 .encode_text)
        class_names: 类别名称列表
        prompt_templates: prompt 模板元组, 用 {label} 填充类别名
        tokenizer: CLIP 分词器
        device: 计算设备

    Returns:
        分类器权重, shape=(num_classes, embed_dim), 已 L2 归一化
    """
    model.eval()

    all_class_embeddings = []

    for class_name in tqdm(class_names, desc="构建零样本分类器", unit="class"):
        # 为当前类别生成所有 prompt 文本
        texts = [
            template.replace("{label}", class_name)
            for template in prompt_templates
        ]

        # tokenize
        tokens = tokenizer.tokenize(texts).to(device)

        # 编码文本并取平均
        text_features = model.encode_text(tokens)  # (num_templates, D)
        class_embedding = text_features.mean(dim=0, keepdim=True)
        class_embedding = F.normalize(class_embedding, dim=-1)
        all_class_embeddings.append(class_embedding)

    classifier = torch.cat(all_class_embeddings, dim=0)  # (num_classes, D)
    return classifier


@torch.no_grad()
def evaluate_dataset(
    model: CLIP,
    dataset_name: str,
    classifier: torch.Tensor,
    class_names: List[str],
    batch_size: int,
    device: torch.device,
    image_resolution: int = 224,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    在指定数据集上进行零样本评估。

    Args:
        model: CLIP 模型
        dataset_name: 数据集名称 ("cifar10" / "cifar100")
        classifier: 零样本分类器权重 (已 L2 归一化)
        class_names: 类别名称列表
        batch_size: 评估 batch size
        device: 计算设备
        image_resolution: 图像输入尺寸
        max_samples: 限制评估样本数 (None 则全量)

    Returns:
        {"top1": float, "top5": float}
    """
    _, dataset_cls = get_dataset_info(dataset_name)

    # 构建预处理 (与训练时保持一致)
    eval_transform = transforms.Compose([
        transforms.Resize(image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_resolution),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    # 加载测试集
    test_dataset = dataset_cls(
        root="data",
        train=False,
        download=True,
        transform=eval_transform,
    )

    if max_samples is not None:
        import random
        indices = list(range(len(test_dataset)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 将分类器移到 device
    classifier = classifier.to(device)

    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc=f"评估 {dataset_name}", unit="batch"):
        images = images.to(device)
        labels = labels.to(device)

        # 编码图像
        image_features = model.encode_image(images)  # (B, D)

        # 计算相似度
        logits = image_features @ classifier.T  # (B, num_classes)

        # Top-1
        pred = logits.argmax(dim=-1)
        top1_correct += (pred == labels).sum().item()

        # Top-5
        _, top5_idx = logits.topk(k=min(5, classifier.shape[0]), dim=-1)
        top5_correct += sum(
            labels[i].item() in top5_idx[i].tolist() for i in range(len(labels))
        )

        total += labels.size(0)

    top1_acc = top1_correct / max(total, 1) * 100
    top5_acc = top5_correct / max(total, 1) * 100

    return {"top1": top1_acc, "top5": top5_acc}


# ============================================================
# 评估主函数
# ============================================================

def evaluate(
    config: Optional[CLIPConfig] = None,
    checkpoint_path: Optional[str] = None,
    eval_datasets: Optional[Tuple[str, ...]] = None,
    prompt_templates: Optional[Tuple[str, ...]] = None,
    batch_size: Optional[int] = None,
    max_samples: Optional[int] = None,
):
    """
    CLIP 零样本分类评估。

    Args:
        config: CLIP 模型配置
        checkpoint_path: 模型权重路径
        eval_datasets: 评估数据集列表
        prompt_templates: prompt 模板
        batch_size: 评估 batch size
        max_samples: 限制样本数 (加快测试)
    """
    if config is None:
        config = CLIPConfig()

    if checkpoint_path is None:
        print("警告: 未提供 checkpoint, 将使用随机初始化模型评估。")

    if eval_datasets is None:
        eval_config = EvalConfig()
        eval_datasets = eval_config.eval_datasets

    if prompt_templates is None:
        eval_config = EvalConfig()
        prompt_templates = eval_config.prompt_templates

    if batch_size is None:
        batch_size = EvalConfig().batch_size

    # ============================================================
    # 设备
    # ============================================================
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ============================================================
    # 构建模型
    # ============================================================
    print("\n构建 CLIP 模型...")
    model = build_clip(config).to(device)
    num_params = model.count_parameters()
    print(f"可训练参数: {num_params:,}")

    # 加载权重
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        print(f"\n加载 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            step = checkpoint.get("step", "?")
            epoch = checkpoint.get("epoch", "?")
            train_loss = checkpoint.get("loss", "?")
            print(f"  训练步数: {step}, epoch: {epoch}, loss: {train_loss}")
        else:
            state_dict = checkpoint

        # 尝试加载 (忽略不匹配的键)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  缺失键: {len(missing)} 个 (投影层等)")
        if unexpected:
            print(f"  多余键: {len(unexpected)} 个")
    elif checkpoint_path is not None:
        print(f"警告: checkpoint 不存在 ({checkpoint_path}), 使用随机初始化。")

    model.eval()

    # ============================================================
    # Tokenizer
    # ============================================================
    from my_clip.tokenizer import CLIPTokenizer
    tokenizer = CLIPTokenizer(context_length=config.context_length)

    # ============================================================
    # 逐个数据集评估
    # ============================================================
    print(f"\n{'='*60}")
    print("零样本分类评估")
    print(f"{'='*60}")
    print(f"评估数据集: {eval_datasets}")
    print(f"Prompt 模板: {prompt_templates}")
    print()

    results = {}

    for dataset_name in eval_datasets:
        print(f"\n{'─'*40}")
        print(f"评估: {dataset_name}")
        print(f"{'─'*40}")

        class_names, _ = get_dataset_info(dataset_name)
        print(f"  类别数: {len(class_names)}")

        # 构建零样本分类器
        print("  构建零样本分类器...")
        classifier = build_zero_shot_classifier(
            model=model,
            class_names=class_names,
            prompt_templates=prompt_templates,
            tokenizer=tokenizer,
            device=device,
        )
        print(f"  分类器 shape: {classifier.shape}")

        # 评估
        print(f"  开始评估 (batch_size={batch_size})...")
        metrics = evaluate_dataset(
            model=model,
            dataset_name=dataset_name,
            classifier=classifier,
            class_names=class_names,
            batch_size=batch_size,
            device=device,
            image_resolution=config.image_resolution,
            max_samples=max_samples,
        )

        results[dataset_name] = metrics
        print(f"\n  {dataset_name.upper()} 结果:")
        print(f"    Top-1 准确率: {metrics['top1']:.2f}%")
        print(f"    Top-5 准确率: {metrics['top5']:.2f}%")

    # ============================================================
    # 汇总结果
    # ============================================================
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("评估汇总")
        print(f"{'='*60}")
        print(f"{'数据集':<12} {'Top-1':>8} {'Top-5':>8}")
        print(f"{'─'*30}")
        for ds, m in results.items():
            print(f"{ds:<12} {m['top1']:>7.2f}% {m['top5']:>7.2f}%")

    return results


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP 零样本评估脚本")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="评估数据集列表, 如: cifar10 cifar100",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="评估 batch size",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="限制评估样本数 (快速测试)",
    )
    args = parser.parse_args()

    eval_datasets = None
    if args.datasets is not None:
        eval_datasets = tuple(args.datasets)

    evaluate(
        checkpoint_path=args.checkpoint,
        eval_datasets=eval_datasets,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
