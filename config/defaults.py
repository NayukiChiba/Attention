"""
config/defaults.py

默认超参数配置
"""

from dataclasses import dataclass, field, fields
from typing import Literal, Tuple

# ============================================================
# 内部工具
# ============================================================


def _format_fields(instance: object, indent: int = 0) -> str:
    """
    递归格式化 dataclass 实例的所有字段

    Args:
        instance: dataclass 实例
        indent: 缩进层级

    Returns:
        str: 格式化后的字符串
    """
    prefix = "  " * indent
    lines = []

    for f in fields(instance):
        value = getattr(instance, f.name)

        # 递归处理嵌套的 dataclass
        if hasattr(value, "__dataclass_fields__"):
            lines.append(f"{prefix}{f.name}:")
            lines.append(_format_fields(value, indent + 1))
        else:
            lines.append(f"{prefix}{f.name} = {value!r}")

    return "\n".join(lines)


# ============================================================
# 模型配置
# ============================================================


@dataclass
class GPTConfig:
    """GPT 模型超参数"""

    # 词表大小（由 tokenizer 运行时确定）
    vocab_size: int = 8000
    # 上下文长度
    block_size: int = 256
    # 模型维度
    d_model: int = 384
    # 注意力头数
    num_heads: int = 8
    # Transformer 层数
    num_layers: int = 8
    # 前馈网络中间维度
    d_ff: int = 1536  # d_model × 4
    # Dropout 比例
    dropout_rate: float = 0.1
    # 注意力 Dropout 比例
    attn_dropout_rate: float = 0.1
    # Embedding Dropout 比例
    embd_dropout_rate: float = 0.1
    # 位置编码类型
    pos_encoding_type: Literal["sinusoidal", "learnable"] = "sinusoidal"
    # 激活函数
    activation: Literal["gelu", "relu"] = "gelu"
    # Norm 类型
    norm_type: Literal["pre", "post"] = "pre"
    # LayerNorm epsilon
    layer_norm_eps: float = 1e-5
    # Embedding 与 lm_head 是否共享权重
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
        assert 0 <= self.attn_dropout_rate < 1, "attn_dropout_rate 必须在 [0, 1) 内"
        assert 0 <= self.embd_dropout_rate < 1, "embd_dropout_rate 必须在 [0, 1) 内"
        assert self.layer_norm_eps > 0, "layer_norm_eps 必须 > 0"

    def _summary(self) -> None:
        """打印当前配置的所有参数"""
        print(f"[{self.__class__.__name__}]")
        print(_format_fields(self))


# ============================================================
# 数据配置
# ============================================================


@dataclass
class TokenizerConfig:
    """分词器配置"""

    # 特殊 token 名称
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    # 是否转为小写
    lowercase: bool = False
    # 最少字符出现次数（低于此频率的字符映射为 UNK）
    min_freq: int = 1
    # 词表保存文件名
    vocab_filename: str = "vocab.json"

    def __post_init__(self):
        tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        assert len(tokens) == len(set(tokens)), "特殊 token 名称不能重复"
        assert self.min_freq >= 1, "min_freq 必须 >= 1"

    def _summary(self) -> None:
        print(f"[{self.__class__.__name__}]")
        print(_format_fields(self))


@dataclass
class DataConfig:
    """数据处理超参数"""

    # 训练集比例
    train_ratio: float = 0.8
    # 验证集比例
    val_ratio: float = 0.1
    # 测试集比例
    test_ratio: float = 0.1
    # 随机种子
    seed: int = 42
    # 最小文本长度（过短的新闻丢弃）
    min_text_length: int = 10
    # 最大文本长度（超过则截断）
    max_text_length: int = 2000
    # 文本清洗：是否合并多余空白
    normalize_whitespace: bool = True
    # 文本清洗：是否移除不可见字符
    remove_control_chars: bool = True
    # 数据集文件编码
    encoding: str = "utf-8"

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, (
            f"train_ratio + val_ratio + test_ratio 必须等于 1，当前: {total}"
        )
        assert self.train_ratio > 0, "train_ratio 必须 > 0"
        assert self.val_ratio >= 0, "val_ratio 必须 >= 0"
        assert self.test_ratio >= 0, "test_ratio 必须 >= 0"
        assert self.min_text_length >= 0, "min_text_length 必须 >= 0"
        assert self.max_text_length > self.min_text_length, (
            f"max_text_length ({self.max_text_length}) 必须 > min_text_length ({self.min_text_length})"
        )

    def _summary(self) -> None:
        print(f"[{self.__class__.__name__}]")
        print(_format_fields(self))


# ============================================================
# 训练配置
# ============================================================


@dataclass
class OptimizerConfig:
    """优化器配置"""

    # 优化器类型
    optimizer_type: Literal["adamw", "sgd"] = "adamw"
    # 基础学习率
    learning_rate: float = 3e-4
    # 权重衰减
    weight_decay: float = 0.01
    # Adam β 参数 (β1, β2)
    betas: Tuple[float, float] = (0.9, 0.999)
    # Adam epsilon（数值稳定性）
    eps: float = 1e-8
    # 是否使用 AMSGrad 变体
    amsgrad: bool = False
    # 是否对 embedding / LayerNorm 参数禁用 weight_decay
    no_decay_bias_norm: bool = True
    # 不施加 weight_decay 的参数名关键词
    no_decay_keywords: Tuple[str, ...] = (
        "bias",
        "layer_norm",
        "layernorm",
        "ln_",
        "embedding",
    )
    # SGD 专用：动量
    momentum: float = 0.9

    def __post_init__(self):
        assert self.learning_rate > 0, "learning_rate 必须 > 0"
        assert self.weight_decay >= 0, "weight_decay 必须 >= 0"
        assert len(self.betas) == 2, "betas 必须是长度为 2 的元组"
        assert 0 < self.betas[0] < 1, f"betas[0] ({self.betas[0]}) 必须在 (0, 1) 内"
        assert 0 < self.betas[1] < 1, f"betas[1] ({self.betas[1]}) 必须在 (0, 1) 内"
        assert self.eps > 0, "eps 必须 > 0"
        assert 0 <= self.momentum < 1, "momentum 必须在 [0, 1) 内"

    def _summary(self) -> None:
        print(f"[{self.__class__.__name__}]")
        print(_format_fields(self))


@dataclass
class SchedulerConfig:
    """学习率调度器配置"""

    # 调度器类型
    scheduler_type: Literal[
        "cosine", "linear", "cosine_warmup", "linear_warmup", "constant"
    ] = "cosine_warmup"
    # 预热步数
    warmup_steps: int = 500
    # 预热起始因子（相对 lr 的比例）
    warmup_start_factor: float = 0.0
    # 总训练步数
    total_steps: int = 10000
    # 最小学习率（衰减终点，相对 lr 的比例）
    min_lr_ratio: float = 0.01
    # 余弦衰减的周期数（1 为单周期）
    cosine_cycles: float = 1.0
    # 达到最小学习率后是否保持
    hold_min_lr: bool = False

    def __post_init__(self):
        assert self.warmup_steps >= 0, "warmup_steps 必须 >= 0"
        assert self.total_steps > 0, "total_steps 必须 > 0"
        assert self.warmup_steps <= self.total_steps, (
            f"warmup_steps ({self.warmup_steps}) 必须 <= total_steps ({self.total_steps})"
        )
        assert 0 <= self.warmup_start_factor <= 1, (
            "warmup_start_factor 必须在 [0, 1] 内"
        )
        assert 0 < self.min_lr_ratio <= 1, "min_lr_ratio 必须在 (0, 1] 内"
        assert self.cosine_cycles > 0, "cosine_cycles 必须 > 0"

    def _summary(self) -> None:
        print(f"[{self.__class__.__name__}]")
        print(_format_fields(self))


@dataclass
class EarlyStoppingConfig:
    """早停配置"""

    # 监控指标
    monitor: Literal["val_loss", "train_loss"] = "val_loss"
    # 耐心值（连续无改善的评估次数）
    patience: int = 5
    # 最小改善幅度（小于此值视为无改善）
    min_delta: float = 0.0
    # 最小值模式（True = loss 越小越好）
    minimize: bool = True
    # 是否恢复到最佳模型权重
    restore_best_weights: bool = False
    # 最佳模型权重保存路径
    best_model_path: str = ""

    def __post_init__(self):
        assert self.patience >= 0, "patience 必须 >= 0"
        assert self.min_delta >= 0, "min_delta 必须 >= 0"

    def _summary(self) -> None:
        print(f"[{self.__class__.__name__}]")
        print(_format_fields(self))


@dataclass
class CheckpointConfig:
    """Checkpoint 配置"""

    # 保存间隔（步数）
    save_interval: int = 1000
    # 是否保存优化器状态
    save_optimizer: bool = True
    # 是否保存调度器状态
    save_scheduler: bool = True
    # 保留最近 N 个 checkpoint（0 为保留全部）
    keep_last_n: int = 3
    # checkpoint 文件名前缀
    prefix: str = "ckpt"
    # checkpoint 文件后缀
    suffix: str = ".pt"
    # 是否压缩保存
    compress: bool = False

    def __post_init__(self):
        assert self.save_interval > 0, "save_interval 必须 > 0"
        assert self.keep_last_n >= 0, "keep_last_n 必须 >= 0"

    def _summary(self) -> None:
        print(f"[{self.__class__.__name__}]")
        print(_format_fields(self))


@dataclass
class LoggerConfig:
    """日志配置"""

    # 控制台日志间隔（步数）
    console_interval: int = 100
    # 评估间隔（步数）
    eval_interval: int = 500
    # 是否记录训练 loss
    log_train_loss: bool = True
    # 是否记录学习率
    log_lr: bool = True
    # 是否记录梯度范数
    log_grad_norm: bool = False
    # 是否记录 GPU 显存
    log_gpu_memory: bool = False
    # 日志文件名
    log_filename: str = "training.log"
    # 是否同时打印到控制台
    console_output: bool = True
    # 日志级别
    level: Literal["INFO", "DEBUG", "WARNING"] = "INFO"

    def __post_init__(self):
        assert self.console_interval > 0, "console_interval 必须 > 0"
        assert self.eval_interval > 0, "eval_interval 必须 > 0"

    def _summary(self) -> None:
        print(f"[{self.__class__.__name__}]")
        print(_format_fields(self))


@dataclass
class TrainingConfig:
    """训练总配置（聚合各子配置的常用参数）"""

    # 批次大小
    batch_size: int = 32
    # 最大 epoch 数
    max_epochs: int = 50
    # 梯度裁剪阈值
    grad_clip: float = 1.0
    # 随机种子
    seed: int = 42
    # 设备
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    # 使用混合精度训练
    use_amp: bool = False
    # 数据加载线程数
    num_workers: int = 4
    # 是否在训练前进行 dry run（单步测试）
    dry_run: bool = False

    # 子配置
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)

    def __post_init__(self):
        assert self.batch_size > 0, "batch_size 必须 > 0"
        assert self.max_epochs > 0, "max_epochs 必须 > 0"
        assert self.grad_clip > 0, "grad_clip 必须 > 0"
        assert self.num_workers >= 0, "num_workers 必须 >= 0"

    def _summary(self) -> None:
        """打印当前配置的所有参数（递归打印子配置）"""
        print(f"[{self.__class__.__name__}]")
        print(_format_fields(self))


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
    # Top-P 采样（核采样，累积概率阈值，1.0 为关闭）
    top_p: float = 0.9
    # 重复惩罚系数（1.0 为不惩罚，>1.0 惩罚重复）
    repetition_penalty: float = 1.0
    # 采样的随机种子（0 为不固定）
    seed: int = 0
    # 停止词（生成到这些词时停止）
    stop_tokens: Tuple[str, ...] = ()
    # 是否使用 KV 缓存加速推理
    use_kv_cache: bool = True
    # 是否流式输出
    stream: bool = False

    def __post_init__(self):
        assert self.max_new_tokens > 0, "max_new_tokens 必须 > 0"
        assert self.temperature >= 0, "temperature 必须 >= 0"
        assert self.top_k >= 0, "top_k 必须 >= 0"
        assert 0 <= self.top_p <= 1, "top_p 必须在 [0, 1] 内"
        assert self.repetition_penalty >= 1.0, "repetition_penalty 必须 >= 1.0"

    def _summary(self) -> None:
        """打印当前配置的所有参数"""
        print(f"[{self.__class__.__name__}]")
        print(_format_fields(self))
