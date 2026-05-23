"""
InfoNCE 对比损失 (Contrastive Loss)

CLIP 使用的对称对比损失:
    loss = (CE(logits_per_image, labels) + CE(logits_per_text, labels)) / 2

其中 labels = [0, 1, ..., batch_size-1], 对角线为正样本对。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    CLIP 对比损失 (InfoNCE)。

    对每一行, 正确标签为该行的索引 (对角线为正样本):
        - logits_per_image[i, j]: 图像 i 与文本 j 的相似度
        - 正样本: i == j
        - 负样本: i != j

    Args:
        logits_per_image: 图像→文本相似度矩阵, shape=(batch_size, batch_size)
        logits_per_text:  文本→图像相似度矩阵, shape=(batch_size, batch_size)
        reduction: "mean" (平均) 或 "none" (每个样本单独返回)

    Returns:
        scalar loss (reduction="mean") 或 (batch_size,) loss (reduction="none")

    参考:
        Radford et al., "Learning Transferable Visual Models From Natural
        Language Supervision", ICML 2021.
    """
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)

    # 两个方向的交叉熵
    loss_i = F.cross_entropy(logits_per_image, labels, reduction=reduction)
    loss_t = F.cross_entropy(logits_per_text, labels, reduction=reduction)

    return (loss_i + loss_t) / 2


class ContrastiveLoss(nn.Module):
    """
    InfoNCE 对比损失的 nn.Module 封装。

    用法:
        loss_fn = ContrastiveLoss()
        logits_img, logits_txt = model(image, tokens)
        loss = loss_fn(logits_img, logits_txt)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits_per_image, logits_per_text):
        return contrastive_loss(
            logits_per_image, logits_per_text, reduction=self.reduction
        )


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("InfoNCE 对比损失测试")
    print("=" * 50)

    batch_size = 4
    embed_dim = 512

    # 模拟随机归一化嵌入
    torch.manual_seed(42)
    img_emb = F.normalize(torch.randn(batch_size, embed_dim), dim=1)
    txt_emb = F.normalize(torch.randn(batch_size, embed_dim), dim=1)

    # 无温度缩放
    logits_img = img_emb @ txt_emb.T
    logits_txt = logits_img.T

    loss = contrastive_loss(logits_img, logits_txt)
    print(f"  batch_size: {batch_size}")
    print(f"  logits shape: {logits_img.shape}")
    print(f"  对角相似度: {logits_img.diag().tolist()}")
    print(f"  loss 值: {loss.item():.4f}")

    # 验证: 当相似度矩阵为单位矩阵时 (完美匹配), loss 应该为 0
    perfect_logits = 100 * torch.eye(batch_size)  # 对角极大, 非对角极低
    perfect_loss = contrastive_loss(perfect_logits, perfect_logits.T)
    print(f"  完美匹配 loss: {perfect_loss.item():.6f} (应接近 0)")

    # 验证: 随机情况下的 loss 上限
    print(f"  随机情况 loss 上限 (ln(batch_size)): {torch.tensor(batch_size).log().item():.4f}")

    # 验证每个样本的 loss
    loss_per_sample = contrastive_loss(logits_img, logits_txt, reduction="none")
    print(f"  每样本 loss: {loss_per_sample.tolist()}")

    # 测试 nn.Module 封装
    loss_fn = ContrastiveLoss()
    loss_module = loss_fn(logits_img, logits_txt)
    assert torch.allclose(loss, loss_module), "Module 封装不一致!"
    print(f"  nn.Module 封装 ok")

    print("\n  InfoNCE 损失测试通过! ✓")
