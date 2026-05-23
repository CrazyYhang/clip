"""
Flickr30k 数据集加载器

每张图片对应 5 条英文描述, 返回 (image_tensor, text_tokens) 对。
支持训练/验证/测试划分。
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Flickr30kDataset(Dataset):
    """
    Flickr30k 图文数据集

    数据格式:
        data/flickr30k/
        ├── images/           # JPEG 图片 (约 31783 张)
        └── captions.csv      # image, caption_number, caption

    返回:
        (image, caption) 元组
        - image: torch.Tensor, shape=(3, 224, 224), 归一化的图像张量
        - caption: str, 原始英文描述文本
    """

    def __init__(
        self,
        data_dir: str = "data/flickr30k",
        split: str = "train",
        image_size: int = 224,
        tokenizer=None,
    ):
        """
        Args:
            data_dir: 数据集根目录 (包含 images/ 和 captions.csv)
            split: 数据集划分 ("train" / "val" / "test")
            image_size: 图像 resize 后的尺寸
            tokenizer: 文本分词器 (如 CLIP SimpleTokenizer)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.split = split
        self.image_size = image_size
        self.tokenizer = tokenizer

        # 图像预处理
        self.transform = self._build_transform(image_size)

        # 加载描述
        self.captions = self._load_captions()

        # 加载 train/val/test 划分
        self.image_ids = self._load_split(split)

        # 构建图文对索引 (每张图片 × 5 条描述)
        self.samples = self._build_samples()

    def _build_transform(self, size: int) -> transforms.Compose:
        """构建图像预处理 pipeline"""
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def _load_captions(self) -> Dict[str, List[str]]:
        """
        加载 captions.csv, 返回 {image_name: [caption1, caption2, ...]} 映射。

        captions.csv 格式 (管道符分隔):
            image_name| comment_number| comment
            1000092795.jpg| 0| Two young guys...
            1000092795.jpg| 1| Two young ...
            ...
        """
        csv_path = self.data_dir / "captions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"未找到描述文件 {csv_path}，请先运行 download_flickr30k.py 下载数据集"
            )

        captions: Dict[str, List[str]] = {}

        # 自动检测分隔符：读第一行，看是 | 还是 ,
        with open(csv_path, "r", encoding="utf-8") as f_sample:
            first_line = f_sample.readline()
        delimiter = "|" if "|" in first_line else ","

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)

            # 图片名列和描述列索引
            header = [h.strip() for h in header] if header else []
            img_idx = 0  # 第一列是图片名
            cap_idx = -1  # 最后一列是描述

            for row in reader:
                fields = [col.strip() for col in row]
                if len(fields) < 2:
                    continue
                image_name = fields[img_idx]
                caption = fields[cap_idx]

                if image_name not in captions:
                    captions[image_name] = []
                captions[image_name].append(caption)

        print(f"  已加载 {len(captions)} 张图片的 {sum(len(v) for v in captions.values())} 条描述 "
              f"(分隔符: {'管道符' if delimiter == '|' else '逗号'})")
        return captions

    def _load_split(self, split: str) -> List[str]:
        """
        加载 train/val/test 划分。

        Flickr30k 标准划分为: train(29000), val(1014), test(1000)
        如果没有现成的划分文件，则按比例自动划分。
        """
        split_file = self.data_dir / "train_test_split.txt"

        if split_file.exists():
            # 使用已有的划分文件
            split_images = []
            with open(split_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        name, label = parts[0], parts[1]
                        if split == "train" and label in ("train", "0"):
                            split_images.append(name)
                        elif split == "val" and label in ("val", "dev", "1"):
                            split_images.append(name)
                        elif split == "test" and label in ("test", "2"):
                            split_images.append(name)
            if split_images:
                valid = [n for n in split_images if n in self.captions]
                print(f"  已加载 {split} 划分: {len(valid)} 张图片 (来自 train_test_split.txt)")
                return valid

        # 没有划分文件，按比例自动划分
        all_images = sorted(self.captions.keys())
        n = len(all_images)

        if split == "train":
            selected = all_images[:int(n * 0.9)]
        elif split == "val":
            selected = all_images[int(n * 0.9):int(n * 0.95)]
        else:  # test
            selected = all_images[int(n * 0.95):]

        print(f"  自动划分 {split}: {len(selected)} 张图片 (无 train_test_split.txt)")
        return selected

    def _build_samples(self) -> List[Tuple[str, str]]:
        """构建 (image_name, caption) 对列表"""
        samples = []
        for image_id in self.image_ids:
            if image_id not in self.captions:
                continue
            for caption in self.captions[image_id]:
                samples.append((image_id, caption))
        print(f"  {self.split} 集共 {len(samples)} 条图文对")
        return samples

    def _find_image_path(self, image_name: str) -> Path:
        """在 images/ 目录中查找图片文件 (搜索 .jpg/.png/.jpeg)"""
        name = Path(image_name).stem
        for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
            path = self.images_dir / f"{name}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"未找到图片: {image_name} 在 {self.images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_name, caption = self.samples[idx]

        # 加载并预处理图像
        image_path = self._find_image_path(image_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        return image_tensor, caption

    def tokenize(self, texts: List[str], context_length: int = 77) -> torch.Tensor:
        """将文本列表转换为 token tensor"""
        if self.tokenizer is None:
            raise ValueError("需要提供 tokenizer 才能调用 tokenize()")

        return self.tokenizer(texts, context_length=context_length)


def collate_fn(batch: List[Tuple[torch.Tensor, str]]):
    """
    默认 DataLoader collate 函数。

    返回:
        images: torch.Tensor, shape=(batch_size, 3, 224, 224)
        captions: List[str], 原始文本列表
    """
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    return images, captions


def visualize_sample(dataset: Flickr30kDataset, idx: int = 0):
    """可视化一组图文对 (需要 matplotlib)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装, 跳过可视化")
        return

    image_tensor, caption = dataset[idx]

    # 反归一化 (大概还原, 便于显示)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    img = image_tensor * std[:, None, None] + mean[:, None, None]
    img = img.permute(1, 2, 0).clamp(0, 1).numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(caption[:100] + ("..." if len(caption) > 100 else ""), fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(dataset.data_dir.parent / "sample_viz.png", dpi=150)
    plt.close()
    print(f"  示例图片已保存到 {dataset.data_dir.parent / 'sample_viz.png'}")


if __name__ == "__main__":
    # 快速测试数据加载
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("=" * 50)
    print("Flickr30k 数据加载测试")
    print("=" * 50)

    # 测试训练集
    train_dataset = Flickr30kDataset(split="train")
    print(f"\n训练集大小: {len(train_dataset)}")
    if len(train_dataset) > 0:
        img, cap = train_dataset[0]
        print(f"图像尺寸: {img.shape}")
        print(f"描述示例: {cap[:150]}...")
        visualize_sample(train_dataset, 0)

    # 测试验证集
    val_dataset = Flickr30kDataset(split="val")
    print(f"\n验证集大小: {len(val_dataset)}")

    # 测试测试集
    test_dataset = Flickr30kDataset(split="test")
    print(f"测试集大小: {len(test_dataset)}")
