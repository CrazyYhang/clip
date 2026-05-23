"""
工具函数

包含:
    - plot_loss_curve(): 从 loss_curve.pt 读取数据并绘制 loss 曲线图, 保存为 PNG

用法:
    from my_clip.utils import plot_loss_curve
    plot_loss_curve("experiments/baseline/checkpoint/loss_curve.pt")
"""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 非交互式后端, 避免 GUI 依赖
import matplotlib.pyplot as plt
import torch


def plot_loss_curve(
    loss_curve_path: str,
    save_path: Optional[str] = None,
    title: str = "CLIP Training Loss Curve",
    figsize: tuple = (10, 6),
    smoothing: float = 0.0,
) -> str:
    """
    读取 loss_curve.pt 并绘制 loss 曲线图, 保存为 PNG。

    loss_curve.pt 格式: {"steps": List[int], "losses": List[float]}

    Args:
        loss_curve_path: loss_curve.pt 文件路径
        save_path: 输出 PNG 路径, 默认为同目录下的 loss_curve.png
        title: 图表标题
        figsize: 图表尺寸 (宽, 高)
        smoothing: EMA 平滑系数 (0=不平滑, 0.9=平滑)

    Returns:
        保存的 PNG 文件路径
    """
    if not os.path.exists(loss_curve_path):
        raise FileNotFoundError(f"未找到 loss_curve.pt: {loss_curve_path}")

    data = torch.load(loss_curve_path, map_location="cpu", weights_only=False)
    steps = data["steps"]
    losses = data["losses"]

    if save_path is None:
        save_path = os.path.join(os.path.dirname(loss_curve_path), "loss_curve.png")

    plt.figure(figsize=figsize)

    # 原始曲线
    plt.plot(steps, losses, alpha=0.4, linewidth=0.8, color="steelblue", label="Raw Loss")

    # 平滑曲线
    if smoothing > 0:
        smoothed = []
        running = None
        for loss in losses:
            if running is None:
                running = loss
            else:
                running = smoothing * running + (1 - smoothing) * loss
            smoothed.append(running)
        plt.plot(steps, smoothed, linewidth=1.5, color="darkorange",
                 label=f"Smoothed (α={smoothing})")

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Contrastive Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Loss 曲线图已保存到: {save_path}")
    return save_path
