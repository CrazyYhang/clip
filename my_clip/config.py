"""
CLIP 训练超参数配置

集中管理所有超参数，避免魔法数字散落各处。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CLIPConfig:
    """CLIP 模型与训练超参数"""

    # ============================================================
    # 模型架构参数 (参见 plan.md 推荐配置)
    # ============================================================

    # 图像编码器
    image_encoder: str = "resnet50"          # 图像编码器类型: resnet50 / resnet101 / vit
    image_embed_dim: int = 512                # 图像嵌入维度
    image_resolution: int = 224               # 输入图像分辨率

    # 文本编码器
    vocab_size: int = 49408                   # 词表大小 (官方 CLIP BPE 词表)
    text_embed_dim: int = 512                 # 文本嵌入维度 (Transformer width)
    text_layers: int = 6                      # Transformer 层数
    text_heads: int = 8                       # 注意力头数
    context_length: int = 77                  # 最大文本长度

    # 联合嵌入空间
    joint_embed_dim: int = 512                # 图文共享嵌入维度
    logit_scale_init: float = 2.659           # log(1/0.07), 温度参数初始值

    # ============================================================
    # 训练参数
    # ============================================================

    # 数据
    dataset_name: str = "flickr30k"           # 数据集名称
    data_dir: str = "data/flickr30k"          # 数据集路径

    # 优化器
    batch_size: int = 32                      # 批次大小 (根据 GPU 显存调整)
    gradient_accumulation_steps: int = 2      # 梯度累积步数 (等效 batch_size = 32 * 2 = 64)
    learning_rate: float = 5e-5               # 初始学习率
    weight_decay: float = 0.2                 # 权重衰减
    adam_beta1: float = 0.9                   # Adam β₁
    adam_beta2: float = 0.98                  # Adam β₂ (官方用 0.999, 这里用 0.98)
    adam_epsilon: float = 1e-6                # Adam ε

    # 学习率调度
    lr_scheduler: str = "cosine"              # 调度器类型: cosine / linear / const
    warmup_steps: int = 500                   # 预热步数 (学习率从 0 到 lr)
    max_epochs: int = 30                      # 最大训练轮数

    # 训练加速
    use_amp: bool = True                      # 混合精度训练 (torch.cuda.amp)
    num_workers: int = 4                      # DataLoader 工作进程数
    pin_memory: bool = True                   # DataLoader pin_memory

    # 日志与保存
    log_interval: int = 10                    # 每隔 N 步打印日志
    save_interval: int = 1                    # 每隔 N 个 epoch 保存 checkpoint
    checkpoint_dir: str = "experiments/baseline/checkpoint"

    # 硬件
    device: str = "cuda"                      # 训练设备


@dataclass
class EvalConfig:
    """零样本评估配置"""

    model_weights: str = ""                   # 模型权重路径 (留空则用随机初始化)
    eval_datasets: Tuple[str, ...] = ("cifar10", "cifar100")
    batch_size: int = 64

    # 评估时用的 prompt 模板
    prompt_templates: Tuple[str, ...] = (
        "a photo of a {label}.",
        "a picture of a {label}.",
        "an image of a {label}.",
    )
