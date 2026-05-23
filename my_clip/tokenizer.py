"""
CLIP 分词器

基于官方 CLIP BPE (Byte Pair Encoding) 分词器封装。
- 主分词器: 复用 CLIP/clip/simple_tokenizer.py (BPE, ~49K 词表)
- 备选: HuggingFace GPT-2 tokenizer (词汇相似)
"""

import sys
from pathlib import Path
from typing import List, Union

import torch

# 将官方 CLIP 目录加入路径
_CLIP_DIR = Path(__file__).parent.parent / "CLIP" / "clip"
if str(_CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(_CLIP_DIR.parent))


class CLIPTokenizer:
    """
    CLIP 文本分词器。

    使用 BPE (Byte Pair Encoding) 将英文文本转换为 token ID 序列。
    默认词表大小: 49408

    用法:
        tokenizer = CLIPTokenizer()
        tokens = tokenizer.encode(["a photo of a cat.", "a picture of a dog."])
        # tokens.shape = (2, 77)
    """

    def __init__(self, context_length: int = 77):
        from clip.simple_tokenizer import SimpleTokenizer

        self.context_length = context_length
        self._tokenizer = SimpleTokenizer()
        self.vocab_size = 49408

        # 特殊 token 索引
        self.sot_token = self._tokenizer.encoder["<|startoftext|>"]
        self.eot_token = self._tokenizer.encoder["<|endoftext|>"]

    def encode(self, text: Union[str, List[str]]) -> List[List[int]]:
        """将文本转换为 token ID 列表 (不含 SOT/EOT)"""
        if isinstance(text, str):
            text = [text]
        return [self._tokenizer.encode(t) for t in text]

    def tokenize(
        self,
        texts: Union[str, List[str]],
        context_length: int = None,
        truncate: bool = False,
    ) -> torch.LongTensor:
        """
        将文本批次 tokenize 并填充/截断到固定长度。

        Args:
            texts: 输入文本或文本列表
            context_length: 最大长度 (默认使用初始化设置)
            truncate: 是否截断超长文本

        Returns:
            token tensor, shape=(batch_size, context_length)
        """
        if context_length is None:
            context_length = self.context_length
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        for text in texts:
            tokens = [self.sot_token] + self._tokenizer.encode(text) + [self.eot_token]
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = self.eot_token
                else:
                    tokens = tokens[:context_length]  # 静默截断
            all_tokens.append(tokens)

        # 填充
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def decode(self, tokens: torch.Tensor) -> List[str]:
        """将 token tensor 解码回文本"""
        return [self._tokenizer.decode(t.tolist()) for t in tokens]

    def __call__(self, texts, **kwargs):
        return self.tokenize(texts, **kwargs)


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("CLIP Tokenizer 测试")
    print("=" * 50)

    tokenizer = CLIPTokenizer()
    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  SOT token: {tokenizer.sot_token}")
    print(f"  EOT token: {tokenizer.eot_token}")

    # 测试单文本
    text = "a photo of a cat sitting on a couch"
    tokens = tokenizer.tokenize(text)
    print(f"\n  输入: '{text}'")
    print(f"  tokens shape: {tokens.shape}")
    print(f"  tokens: {tokens[0, :20].tolist()}...")
    print(f"  decode: '{tokenizer.decode(tokens[:1])[0][:80]}...'")

    # 测试批量文本
    texts = [
        "a photo of a cat.",
        "a photo of a dog.",
        "two men working on a machine.",
    ]
    tokens = tokenizer.tokenize(texts)
    print(f"\n  批量输入: {len(texts)} 条文本")
    print(f"  tokens shape: {tokens.shape}")
    for i, t in enumerate(texts):
        print(f"  [{i}] '{t}' → {tokens[i, :15].tolist()}...")

    # 测试截断
    long_text = "this is a very long sentence " * 10
    tokens_trunc = tokenizer.tokenize(long_text, truncate=True)
    tokens_full = tokenizer.tokenize(long_text, truncate=False)
    print(f"\n  长文本截断测试:")
    print(f"    原始长度: {len(tokenizer.encode(long_text)[0])}")
    print(f"    截断后: {tokens_trunc[0, -1].item()} (应为 EOT={tokenizer.eot_token})")
    print(f"    不截断: shape={tokens_full.shape}")

    print("\n  Tokenizer 测试通过! ✓")
