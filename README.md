# CLIP 论文复现与改进

基于 PyTorch 实现的 CLIP (Contrastive Language-Image Pre-training) 模型复现项目。

参考论文：[Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021](https://arxiv.org/abs/2103.00020)

## 项目结构

```
.
├── my_clip/                 # 自实现 CLIP 核心代码
│   ├── model.py             # 双编码器模型 (ResNet-50 + Transformer)
│   ├── train.py             # 训练脚本 (AMP、梯度累积、cosine 调度)
│   ├── eval.py              # 零样本分类评估 (CIFAR-10 / CIFAR-100)
│   ├── loss.py              # InfoNCE 对比损失
│   ├── config.py            # 超参数配置
│   ├── dataset.py           # Flickr30k 数据集加载
│   ├── tokenizer.py         # BPE 分词器
│   └── utils.py             # 可视化工具
├── CLIP/                    # OpenAI 官方 CLIP 源码 (参考)
└── download_flickr30k.py    # Flickr30k 下载脚本
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- torchvision
- tensorboard
- tqdm
- matplotlib

```bash
pip install torch torchvision tensorboard tqdm matplotlib
```

### 训练

```bash
# 使用默认配置训练
python -m my_clip.train

# 自定义参数
python -m my_clip.train --epochs 10 --batch-size 64 --lr 1e-4

# 快速测试 (限制步数)
python -m my_clip.train --max-steps 100
```

### 零样本评估

```bash
python -m my_clip.eval --checkpoint experiments/baseline/checkpoint/best_model.pt
```

## 模型架构

| 组件 | 配置 |
|------|------|
| 图像编码器 | ResNet-50 (torchvision 预训练) + 投影层 |
| 文本编码器 | 6 层 Transformer, 8 头, 嵌入维度 512 |
| 联合空间 | 512 维 L2 归一化嵌入 |
| 分词器 | BPE, 词表大小 49408, 最大长度 77 |
| 损失函数 | 对称 InfoNCE |
| 训练数据 | Flickr30k |

## 参考

- 原始论文及官方代码：[openai/CLIP](https://github.com/openai/CLIP)
