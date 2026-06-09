"""
config/defaults.py

默认超参数配置
"""

from dataclasses import dataclass, fields
from typing import Literal, Tuple

import torch

# ============================================================
# 内部工具
# ============================================================


def _summary(instance: object) -> None:
    """打印 dataclass 实例的所有参数（私有工具函数）"""
    print(f"[{instance.__class__.__name__}]")
    for f in fields(instance):
        value = getattr(instance, f.name)
        print(f"  {f.name} = {value!r}")


# ============================================================
# 数据配置
# ============================================================


@dataclass
class DataConfig:
    """数据处理 & 分词器超参数"""

    # === 特殊 token ===
    # 填充 token
    pad_token: str = "<PAD>"
    # 未知字符 token
    unk_token: str = "<UNK>"
    # 序列起始 token
    bos_token: str = "<BOS>"
    # 序列结束 token
    eos_token: str = "<EOS>"

    # === 数据集划分 ===
    # 训练集占比
    train_ratio: float = 0.8
    # 验证集占比
    val_ratio: float = 0.1
    # 测试集占比
    test_ratio: float = 0.1
    # 随机种子（保证划分可复现）
    seed: int = 42

    # === 文本过滤 ===
    # 最短文本长度（低于此值的新闻丢弃）
    min_text_length: int = 10
    # 最长文本长度（超过则截断）
    max_text_length: int = 2000

    def __post_init__(self):
        tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        assert len(tokens) == len(set(tokens)), "特殊 token 名称不能重复"

        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, "train/val/test 比例之和必须为 1"
        assert self.train_ratio > 0, "train_ratio 必须 > 0"
        assert self.min_text_length >= 0, "min_text_length 必须 >= 0"
        assert self.max_text_length > self.min_text_length, (
            "max_text_length 必须 > min_text_length"
        )

    def _summary(self) -> None:
        _summary(self)


# ============================================================
# 模型配置
# ============================================================


@dataclass
class GPTConfig:
    """GPT 模型结构超参数"""

    # 词表大小（运行时由 tokenizer 确定）
    vocab_size: int = 8000
    # 上下文长度（最大 token 数）
    block_size: int = 256
    # 模型隐藏维度
    d_model: int = 384
    # 多头注意力头数
    num_heads: int = 8
    # Transformer 解码器层数
    num_layers: int = 8
    # 前馈网络中间维度
    d_ff: int = 1536  # d_model × 4
    # Dropout 比例
    dropout_rate: float = 0.1
    # 位置编码类型
    pos_encoding_type: Literal["sinusoidal", "learnable"] = "sinusoidal"
    # 激活函数类型
    activation: Literal["gelu", "relu"] = "gelu"
    # LayerNorm 位置（pre-norm 训练更稳定）
    norm_type: Literal["pre", "post"] = "pre"
    # LayerNorm 数值稳定项
    layer_norm_epsilon: float = 1e-5
    # 输出层与词嵌入层共享权重（减少参数量）
    tie_weights: bool = True

    def __post_init__(self):
        assert self.vocab_size > 0, "vocab_size 必须 > 0"
        assert self.block_size > 0, "block_size 必须 > 0"
        assert self.d_model > 0, "d_model 必须 > 0"
        assert self.num_heads > 0, "num_heads 必须 > 0"
        assert self.num_layers > 0, "num_layers 必须 > 0"
        assert self.d_ff > 0, "d_ff 必须 > 0"
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) 必须能被 num_heads ({self.num_heads}) 整除"
        )
        assert 0 <= self.dropout_rate < 1, "dropout_rate 必须在 [0, 1) 内"
        assert self.layer_norm_epsilon > 0, "layer_norm_epsilon 必须 > 0"

    def _summary(self) -> None:
        _summary(self)


# ============================================================
# 训练配置
# ============================================================


@dataclass
class TrainingConfig:
    """训练超参数（优化器、调度器、早停、checkpoint、日志 全部扁平）"""

    # === 训练循环 ===
    # 批次大小
    batch_size: int = 32
    # 最大训练轮数
    max_epochs: int = 50
    # 总训练步数
    total_steps: int = 10000
    # 梯度裁剪阈值
    grad_clip: float = 1.0
    # 随机种子（保证训练可复现）
    seed: int = 42
    # 训练设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # DataLoader 线程数
    num_workers: int = 4

    # === 优化器 ===
    # 优化器类型
    optimizer_type: Literal["adam", "adamw", "sgd"] = "adamw"
    # 初始学习率
    learning_rate: float = 3e-4
    # 权重衰减系数
    weight_decay: float = 0.01
    # Adam 动量参数 (β1, β2)
    betas: Tuple[float, float] = (0.9, 0.999)
    # 数值稳定项
    eps: float = 1e-8

    # === 学习率调度 ===
    # 调度策略
    scheduler_type: Literal["cosine_warmup", "constant"] = "cosine_warmup"
    # 预热步数（线性从 0 升至 learning_rate）
    warmup_steps: int = 500
    # 最低学习率（相对 learning_rate 的比例）
    min_lr_ratio: float = 0.01

    # === 早停 ===
    # 容忍轮数（连续无改善则停止）
    early_stopping_patience: int = 5
    # 最小改善阈值
    early_stopping_min_delta: float = 0.0

    def __post_init__(self):
        assert self.batch_size > 0, "batch_size 必须 > 0"
        assert self.max_epochs > 0, "max_epochs 必须 > 0"
        assert self.total_steps > 0, "total_steps 必须 > 0"
        assert self.grad_clip > 0, "grad_clip 必须 > 0"
        assert self.num_workers >= 0, "num_workers 必须 >= 0"

        assert self.learning_rate > 0, "learning_rate 必须 > 0"
        assert self.weight_decay >= 0, "weight_decay 必须 >= 0"
        assert 0 < self.betas[0] < 1, f"betas[0] ({self.betas[0]}) 必须在 (0, 1) 内"
        assert 0 < self.betas[1] < 1, f"betas[1] ({self.betas[1]}) 必须在 (0, 1) 内"
        assert self.eps > 0, "eps 必须 > 0"

        assert self.warmup_steps >= 0, "warmup_steps 必须 >= 0"
        assert self.warmup_steps <= self.total_steps, "warmup_steps 必须 <= total_steps"
        assert 0 < self.min_lr_ratio <= 1, "min_lr_ratio 必须在 (0, 1] 内"

        assert self.early_stopping_patience >= 0, "early_stopping_patience 必须 >= 0"

    def _summary(self) -> None:
        _summary(self)


# ============================================================
# 推理配置
# ============================================================


@dataclass
class GenerationConfig:
    """文本生成超参数"""

    # 最大生成 token 数
    max_new_tokens: int = 200
    # 温度（越高越随机，0 为贪心解码）
    temperature: float = 0.8
    # Top-K 采样（保留概率最高的 K 个 token，0 为关闭）
    top_k: int = 50
    # Top-P 核采样（累积概率阈值，1.0 为关闭）
    top_p: float = 0.9
    # 重复惩罚（1.0 不惩罚，>1.0 抑制重复）
    repetition_penalty: float = 1.0
    # 是否使用 KV 缓存加速推理
    use_kv_cache: bool = True

    def __post_init__(self):
        assert self.max_new_tokens > 0, "max_new_tokens 必须 > 0"
        assert self.temperature >= 0, "temperature 必须 >= 0"
        assert self.top_k >= 0, "top_k 必须 >= 0"
        assert 0 <= self.top_p <= 1, "top_p 必须在 [0, 1] 内"
        assert self.repetition_penalty >= 1.0, "repetition_penalty 必须 >= 1.0"

    def _summary(self) -> None:
        _summary(self)
