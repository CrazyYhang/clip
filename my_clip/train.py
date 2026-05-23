"""
CLIP 训练脚本

包含:
    - AMP 混合精度训练 (torch.cuda.amp)
    - 梯度累积 (模拟大 batch_size)
    - Cosine 学习率调度 + 线性 warmup
    - 定期保存 checkpoint
    - TensorBoard 日志记录
    - 训练 loss 曲线数据记录

用法:
    python -m my_clip.train                         # 使用默认配置
    python -m my_clip.train --epochs 10              # 覆盖训练轮数
    python -m my_clip.train --max-steps 100          # 限制步数 (快速测试)
"""

import argparse
import math
import os
import sys
import time

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 确保 my_clip 在 Python path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from my_clip.config import CLIPConfig
from my_clip.dataset import Flickr30kDataset
from my_clip.loss import ContrastiveLoss
from my_clip.model import build_clip
from my_clip.tokenizer import CLIPTokenizer


# ============================================================
# Collate 函数工厂
# ============================================================

def make_collate_fn(tokenizer: CLIPTokenizer, context_length: int = 77):
    """
    创建支持 tokenize 的 collate 函数。

    Dataset 返回 (image_tensor, caption_str), 需要将文本转换为 token。
    """

    def collate_fn(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item[0] for item in batch])
        texts = [item[1] for item in batch]
        tokens = tokenizer.tokenize(texts, context_length=context_length)
        return images, tokens

    return collate_fn


# ============================================================
# 学习率调度器
# ============================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    创建 cosine 学习率调度器 (含线性 warmup)。

    学习率变化:
        0 → warmup_steps: 线性从 0 增加到 lr
        warmup_steps → total_steps: cosine 衰减到 0
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Checkpoint 保存 / 加载
# ============================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    scaler: torch.amp.GradScaler,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: str,
    config: CLIPConfig,
    filename: str = "checkpoint.pt",
):
    """保存训练 checkpoint。"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
        "config": config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def save_loss_curve(losses: List[float], steps: List[int], save_path: str):
    """保存 loss 曲线数据 (兼容 matplotlib 绘图)。"""
    torch.save({"steps": steps, "losses": losses}, save_path)


# ============================================================
# 训练主函数
# ============================================================

def train(
    config: Optional[CLIPConfig] = None,
    max_steps: Optional[int] = None,
):
    """
    CLIP 模型训练。

    Args:
        config: CLIPConfig 配置实例 (None 则用默认配置)
        max_steps: 最大训练步数, 覆盖 max_epochs (用于快速测试)
    """
    if config is None:
        config = CLIPConfig()

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

    # ============================================================
    # Tokenizer
    # ============================================================
    print("\n初始化 tokenizer...")
    tokenizer = CLIPTokenizer(context_length=config.context_length)
    print(f"词表大小: {tokenizer.vocab_size}")

    # ============================================================
    # 数据集
    # ============================================================
    print(f"\n加载 Flickr30k 数据集 (train)...")
    train_dataset = Flickr30kDataset(
        data_dir=config.data_dir,
        split="train",
        image_size=config.image_resolution,
    )
    print(f"训练样本数: {len(train_dataset)}")

    # ============================================================
    # DataLoader
    # ============================================================
    collate_fn = make_collate_fn(tokenizer, context_length=config.context_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,  # 丢弃不完整的最后一批 (避免 batch_size=1 时 loss 异常)
    )

    # ============================================================
    # 损失函数
    # ============================================================
    loss_fn = ContrastiveLoss()

    # ============================================================
    # 优化器 (AdamW, 按 CLIP 官方论文)
    # ============================================================
    # 排除 gain 和 bias 项的 weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "ln" in name or "layernorm" in name or "gain" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )

    # ============================================================
    # 学习率调度器
    # ============================================================
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = config.max_epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = max_steps

    total_steps = max(total_steps, 1)  # 防止为 0

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=min(config.warmup_steps, total_steps),
        total_steps=total_steps,
    )

    print(f"\n训练配置:")
    print(f"  batch_size: {config.batch_size}")
    print(f"  梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"  等效 batch_size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  warmup_steps: {min(config.warmup_steps, total_steps)}")
    print(f"  total_steps: {total_steps}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  weight_decay: {config.weight_decay}")
    print(f"  混合精度: {'是' if config.use_amp else '否'}")

    # ============================================================
    # AMP 混合精度
    # ============================================================
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    # ============================================================
    # TensorBoard 日志
    # ============================================================
    log_dir = Path(config.checkpoint_dir).parent / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir), filename_suffix="_clip")

    # ============================================================
    # 训练状态
    # ============================================================
    train_losses: List[float] = []
    train_steps: List[int] = []
    global_step = 0
    best_loss = float("inf")
    start_time = time.time()

    # ============================================================
    # 训练循环
    # ============================================================
    print("\n开始训练...")

    model.train()

    epoch_pbar = tqdm(total=config.max_epochs, desc="Training", unit="epoch")

    for epoch in range(1, config.max_epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0

        step_pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{config.max_epochs}", unit="step", leave=False, dynamic_ncols=True)

        for batch_idx, (images, tokens) in enumerate(train_loader):
            images = images.to(device, non_blocking=config.pin_memory)
            tokens = tokens.to(device, non_blocking=config.pin_memory)

            # 前向传播 (AMP)
            with torch.amp.autocast("cuda", enabled=config.use_amp):
                logits_img, logits_txt = model(images, tokens)
                loss = loss_fn(logits_img, logits_txt)
                loss = loss / config.gradient_accumulation_steps

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度累积: 每 accumulation_steps 步更新一次
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                global_step += 1
                current_loss = loss.item() * config.gradient_accumulation_steps

                elapsed = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1)
                step_pbar.update(1)
                step_pbar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "τ": f"{model.logit_scale.exp().item():.2f}",
                    "spd": f"{steps_per_sec:.1f}/s",
                })

                # 记录 loss
                train_losses.append(current_loss)
                train_steps.append(global_step)
                epoch_loss += current_loss
                epoch_steps += 1

                # TensorBoard 日志
                writer.add_scalar("Loss/train", current_loss, global_step)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar(
                    "logit_scale", model.logit_scale.exp().item(), global_step
                )

                # 保存最佳 loss, checkpoint
                if current_loss < best_loss:
                    best_loss = current_loss
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        step=global_step,
                        loss=current_loss,
                        checkpoint_dir=config.checkpoint_dir,
                        config=config,
                        filename="best_model.pt",
                    )
                    save_loss_curve(
                        train_losses, train_steps,
                        os.path.join(config.checkpoint_dir, "loss_curve.pt"),
                    )

                # 达到最大步数则提前退出
                if max_steps is not None and global_step >= max_steps:
                    break

        # epoch 结束
        step_pbar.close()
        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss / epoch_steps
            epoch_pbar.set_postfix({"avg_loss": f"{avg_epoch_loss:.4f}"})
        epoch_pbar.update(1)

        # 定期保存 checkpoint
        if epoch % config.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                loss=avg_epoch_loss if epoch_steps > 0 else 0.0,
                checkpoint_dir=config.checkpoint_dir,
                config=config,
                filename=f"checkpoint_epoch_{epoch}.pt",
            )

        # 提前退出
        if max_steps is not None and global_step >= max_steps:
            print(f"\n达到最大步数 {max_steps}, 提前结束训练。")
            break

    epoch_pbar.close()

    # ============================================================
    # 训练结束, 保存最终模型
    # ============================================================
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"训练完成! 总耗时: {elapsed/60:.1f} 分钟")
    print(f"总步数: {global_step}")
    print(f"最佳 loss: {best_loss:.4f}")
    print(f"{'='*60}")

    # 保存最终 checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        step=global_step,
        loss=train_losses[-1] if train_losses else 0.0,
        checkpoint_dir=config.checkpoint_dir,
        config=config,
        filename="final_model.pt",
    )

    # 保存 loss 曲线数据
    save_loss_curve(
        train_losses, train_steps,
        os.path.join(config.checkpoint_dir, "loss_curve.pt"),
    )

    writer.close()
    print(f"TensorBoard 日志已保存到: {log_dir}")
    print(f"Checkpoint 已保存到: {config.checkpoint_dir}")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP 训练脚本")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--max-steps", type=int, default=None, help="最大训练步数 (快速测试)")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--no-amp", action="store_true", help="禁用混合精度")
    parser.add_argument("--data-dir", type=str, default=None, help="数据集路径")
    args = parser.parse_args()

    # 构建配置
    config = CLIPConfig()

    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.no_amp:
        config.use_amp = False
    if args.data_dir is not None:
        config.data_dir = args.data_dir

    train(config=config, max_steps=args.max_steps)
