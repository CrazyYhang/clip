"""
CLIP 模型定义

基于 OpenAI CLIP (2021) 论文实现, 包含:
- 图像编码器: ResNet-50 (torchvision 预训练初始化) + 投影层
- 文本编码器: Transformer + 投影层
- 对比学习: 通过 InfoNCE loss 对齐图文嵌入

参考: CLIP/clip/model.py (官方推理代码)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# 图像编码器
# ============================================================

class ImageEncoder(nn.Module):
    """
    基于 ResNet-50 的图像编码器。

    移除最后的分类头 (fc), 替换为线性投影层到联合嵌入空间。
    支持多种 ResNet 变体 (50/101) 和从头训练 / 预训练初始化。
    """

    def __init__(
        self,
        embed_dim: int = 512,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_bn: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 加载 ResNet backbone
        if backbone == "resnet50":
            resnet = models.resnet50(weights="DEFAULT" if pretrained else None)
            self.feature_dim = 2048
        elif backbone == "resnet101":
            resnet = models.resnet101(weights="DEFAULT" if pretrained else None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"不支持的 backbone: {backbone}")

        # 移除最后的全连接层和池化层 (使用 adaptive_avg_pool)
        modules = list(resnet.children())[:-2]  # 去掉 avgpool 和 fc
        self.backbone = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 投影层: feature_dim → embed_dim
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim, bias=False),
        )

        # 可选冻结 BatchNorm
        if freeze_bn:
            self._freeze_bn()

        self._init_weights()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def _init_weights(self):
        nn.init.normal_(self.projection[0].weight, std=self.feature_dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量, shape=(batch_size, 3, H, W)
        Returns:
            特征向量, shape=(batch_size, embed_dim)
        """
        features = self.backbone(x)               # (B, 2048, 7, 7)
        pooled = self.avgpool(features)            # (B, 2048, 1, 1)
        pooled = pooled.flatten(1)                 # (B, 2048)
        return self.projection(pooled)             # (B, embed_dim)


# ============================================================
# 文本编码器
# ============================================================

class TextEncoder(nn.Module):
    """
    基于 Transformer 的文本编码器。

    使用 nn.TransformerEncoder, 带有 causal attention mask (单向注意力)。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        context_length: int = 77,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.pad_token_id = pad_token_id

        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        # 位置嵌入
        self.positional_embedding = nn.Parameter(
            torch.randn(context_length, embed_dim) * 0.01
        )

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (batch, seq, dim) 便于处理
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最终 LayerNorm
        self.ln_final = nn.LayerNorm(embed_dim)

        # Causal attention mask (上三角为 -inf)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((context_length, context_length), float("-inf")), diagonal=1),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: 文本 token 序列, shape=(batch_size, seq_len)
        Returns:
            文本特征向量, shape=(batch_size, embed_dim)
            取 EOT token 位置的输出 (序列中最大 token_id 的位置)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # token embedding + positional embedding
        x = self.token_embedding(tokens)  # (B, L, D)
        x = x + self.positional_embedding[:seq_len, :]

        # 构建 causal + padding mask
        causal_mask = self.causal_mask[:seq_len, :seq_len].to(device)

        # padding mask: True 表示需要 mask 的位置
        pad_mask = (tokens == self.pad_token_id)

        # Transformer (batch_first=True, 输入 shape = (B, L, D))
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
        )  # (B, L, D)

        # LayerNorm
        x = self.ln_final(x)  # (B, L, D)

        # 取每个序列的最后一个非 pad token 作为文本表示
        # 找到每行的最后一个非零 token 索引
        if pad_mask.all(dim=1).any():
            # 如果某行全是 pad, 取最后一个位置
            lengths = seq_len - pad_mask.sum(dim=1)
            lengths = torch.clamp(lengths, min=1)
        else:
            lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)

        eos_idx = lengths - 1  # (B,)

        # gather the eos token embeddings
        x = x[torch.arange(batch_size, device=device), eos_idx]  # (B, D)

        return x


# ============================================================
# CLIP 模型
# ============================================================

class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-training

    双编码器结构:
        image  → ImageEncoder → normalized embedding
        text   → TextEncoder   → normalized embedding
        logits = logit_scale * image_emb @ text_emb.T

    参考原始论文 (Radford et al., 2021)
    """

    def __init__(
        self,
        # 图像编码器
        image_embed_dim: int = 512,
        image_backbone: str = "resnet50",
        image_pretrained: bool = True,
        # 文本编码器
        vocab_size: int = 49408,
        text_embed_dim: int = 512,
        context_length: int = 77,
        text_layers: int = 6,
        text_heads: int = 8,
        text_ff_dim: int = 2048,
        text_dropout: float = 0.1,
        # 联合空间
        joint_embed_dim: int = 512,
        logit_scale_init: float = 2.659,  # log(1/0.07)
    ):
        super().__init__()

        self.joint_embed_dim = joint_embed_dim

        # 图像编码器
        self.image_encoder = ImageEncoder(
            embed_dim=image_embed_dim if image_embed_dim == joint_embed_dim else image_embed_dim,
            backbone=image_backbone,
            pretrained=image_pretrained,
        )

        # 图像投影层 (当 image_embed_dim ≠ joint_embed_dim 时使用)
        self.image_projection = None
        if image_embed_dim != joint_embed_dim:
            self.image_projection = nn.Linear(image_embed_dim, joint_embed_dim, bias=False)

        # 文本编码器
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            context_length=context_length,
            num_layers=text_layers,
            num_heads=text_heads,
            ff_dim=text_ff_dim,
            dropout=text_dropout,
        )

        # 文本投影层
        self.text_projection = None
        if text_embed_dim != joint_embed_dim:
            self.text_projection = nn.Linear(text_embed_dim, joint_embed_dim, bias=False)

        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

        self._init_projections()

    def _init_projections(self):
        """初始化投影层权重"""
        if self.image_projection is not None:
            nn.init.normal_(self.image_projection.weight, std=self.image_encoder.feature_dim ** -0.5)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection.weight, std=self.text_encoder.embed_dim ** -0.5)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像, 返回 L2 归一化的嵌入"""
        features = self.image_encoder(image)
        if self.image_projection is not None:
            features = self.image_projection(features)
        return F.normalize(features, dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """编码文本, 返回 L2 归一化的嵌入"""
        features = self.text_encoder(tokens)
        if self.text_projection is not None:
            features = self.text_projection(features)
        return F.normalize(features, dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            image: 图像批次, shape=(batch_size, 3, H, W)
            tokens: 文本 token 批次, shape=(batch_size, context_length)

        Returns:
            logits_per_image: (batch_size, batch_size)
            logits_per_text:  (batch_size, batch_size)
        """
        # 编码 + L2 归一化
        image_features = self.encode_image(image)  # (B, D)
        text_features = self.encode_text(tokens)    # (B, D)

        # 计算余弦相似度 (scaled by temperature)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 模型工厂函数
# ============================================================

def build_clip(config=None) -> CLIP:
    """
    根据配置构建 CLIP 模型。

    Args:
        config: CLIPConfig 实例 (可选, 不传则用默认值)
    """
    if config is None:
        from my_clip.config import CLIPConfig
        config = CLIPConfig()

    return CLIP(
        image_embed_dim=config.image_embed_dim,
        image_backbone=config.image_encoder,
        vocab_size=config.vocab_size,
        text_embed_dim=config.text_embed_dim,
        context_length=config.context_length,
        text_layers=config.text_layers,
        text_heads=config.text_heads,
        joint_embed_dim=config.joint_embed_dim,
        logit_scale_init=config.logit_scale_init,
    )


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/mnt/d/nlp")

    print("=" * 50)
    print("CLIP 模型前向传播测试")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    # 构建模型
    model = build_clip().to(device)
    model.train()

    params = model.count_parameters()
    print(f"  可训练参数: {params:,}")

    # 随机输入测试
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_tokens = torch.randint(0, 49408, (batch_size, 77)).to(device)

    with torch.no_grad():
        logits_img, logits_txt = model(dummy_image, dummy_tokens)

    print(f"\n  输入图像: {dummy_image.shape}")
    print(f"  输入 tokens: {dummy_tokens.shape}")
    print(f"  输出 logits_per_image: {logits_img.shape} (期望: (4, 4))")
    print(f"  输出 logits_per_text:  {logits_txt.shape} (期望: (4, 4))")

    assert logits_img.shape == (batch_size, batch_size), f"形状错误! {logits_img.shape}"
    assert logits_txt.shape == (batch_size, batch_size), f"形状错误! {logits_txt.shape}"
    print(f"\n  logit_scale: {model.logit_scale.exp().item():.4f}")
    print(f"  对角相似度: {logits_img.diag().tolist()} (应高于非对角)")

    # 验证  encode_image / encode_text
    img_emb = model.encode_image(dummy_image)
    txt_emb = model.encode_text(dummy_tokens)
    print(f"  图像嵌入: {img_emb.shape}, 范数: {img_emb.norm(dim=1)}")
    print(f"  文本嵌入: {txt_emb.shape}, 范数: {txt_emb.norm(dim=1)}")

    print("\n  模型前向传播通过! ✓")
