# Attention

从零手搓 Attention 模块，纯 PyTorch 张量运算实现 Transformer，训练小型 GPT 语言模型。

## 项目简介

不依赖 `nn.MultiheadAttention`，从 `softmax(QK^T/√dk + mask) @ V` 开始，一步步搭建完整的 GPT 风格语言模型。使用 THUCNews 中文新闻数据集训练，最终生成通顺的中文文本。

**教学优先**：每一步都自己实现，理解 Attention 的每一个细节。

## 快速开始

```bash
# 安装依赖
uv sync

# 训练模型
python main.py train

# 生成文本
python main.py generate --prompt "今天" --length 100

# 评估模型
python main.py evaluate --checkpoint outputs/checkpoints/best.pt
```

## 项目结构

```
Attention/
├── main.py
├── configs/
│   ├── paths.py               # 路径常量
│   └── defaults.py            # 默认超参数
├── src/
│   ├── cli/                   # 命令行接口
│   ├── data/                  # 数据处理（分词、数据集、下载）
│   ├── model/                 # 模型（手搓 Attention 从头搭建）
│   ├── train/                 # 训练模块（子模块零耦合）
│   ├── evaluate/              # 评估（指标、可视化）
│   └── inference/             # 推理（采样策略、文本生成）
├── tests/
├── datasets/
├── outputs/
└── pyproject.toml
```

## 架构说明

**模型搭建**：mask.py + scaled_dot_product_attention.py → multi_head_attention.py → transformer_block.py → gpt.py。每一层都是自己实现的，没有黑盒。

**训练模块**：trainer.py 是唯一调度者，optimizer、scheduler、early_stopping、checkpoint、logger、utils 六个子模块彼此无 import 关系，各自独立可测。

**单一入口**：`main.py train|generate|evaluate` 子命令分发。

## 模型参数

| 参数 | 值 |
|------|-----|
| d_model | 384 |
| num_heads | 8 |
| num_layers | 8 |
| d_ff | 1536 |
| block_size | 256 |
| 总参数量 | ~15M |

## 数据集

[THUCNews](http://thuctc.thunlp.org/) — 清华大学发布，74 万篇中文新闻，14 个类别，UTF-8 格式。

## 依赖

- Python >= 3.11
- PyTorch >= 2.0
- tqdm、matplotlib

## License

MIT
