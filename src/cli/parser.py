"""
src/cli/parser.py

命令行参数解析器

用法:
    python main.py train [options]
    python main.py eval [options]
    python main.py generate [options]
    python main.py               (交互式菜单)
"""

import argparse

from config.defaults import GenerationConfig, GPTConfig, TrainingConfig


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""

    # ============================================================
    # 默认配置值
    # ============================================================
    train_defaults = TrainingConfig()
    model_defaults = GPTConfig()
    gen_defaults = GenerationConfig()

    # ============================================================
    # 主解析器
    # ============================================================
    parser = argparse.ArgumentParser(
        description="GPT 中文新闻文本生成模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ============================================================
    # train 子命令
    # ============================================================
    train_parser = subparsers.add_parser("train", help="训练模型")

    # 模型 & 数据
    train_group_model = train_parser.add_argument_group("模型和数据配置")
    train_group_model.add_argument(
        "--vocab-size",
        type=int,
        default=model_defaults.vocab_size,
        help="词表大小(默认: %(default)s)",
    )
    train_group_model.add_argument(
        "--context-length",
        type=int,
        default=model_defaults.context_length,
        help="上下文长度(默认: %(default)s)",
    )
    train_group_model.add_argument(
        "--embedding-dim",
        type=int,
        default=model_defaults.embedding_dim,
        help="嵌入维度(默认: %(default)s)",
    )
    train_group_model.add_argument(
        "--num-heads",
        type=int,
        default=model_defaults.num_attention_heads,
        help="注意力头数(默认: %(default)s)",
    )
    train_group_model.add_argument(
        "--num-layers",
        type=int,
        default=model_defaults.num_layers,
        help="Transformer 层数(默认: %(default)s)",
    )
    train_group_model.add_argument(
        "--ffn-hidden-dim",
        type=int,
        default=model_defaults.ffn_hidden_dim,
        help="FFN 隐藏层维度(默认: %(default)s)",
    )
    train_group_model.add_argument(
        "--dropout",
        type=float,
        default=model_defaults.dropout_rate,
        help="Dropout 比例(默认: %(default)s)",
    )
    train_group_model.add_argument(
        "--pos-encoding",
        type=str,
        default=model_defaults.pos_encoding_type,
        choices=["sinusoidal", "learnable"],
        help="位置编码类型(默认: %(default)s)",
    )
    train_group_model.add_argument(
        "--activation",
        type=str,
        default=model_defaults.activation,
        choices=["gelu", "relu"],
        help="激活函数(默认: %(default)s)",
    )
    train_group_model.add_argument(
        "--norm-type",
        type=str,
        default=model_defaults.norm_type,
        choices=["pre", "post"],
        help="LayerNorm 位置(默认: %(default)s)",
    )

    # 训练循环
    train_group_train = train_parser.add_argument_group("训练循环配置")
    train_group_train.add_argument(
        "--batch-size",
        type=int,
        default=train_defaults.batch_size,
        help="批次大小(默认: %(default)s)",
    )
    train_group_train.add_argument(
        "--epochs",
        type=int,
        default=train_defaults.max_epochs,
        help="最大训练轮数(默认: %(default)s)",
    )
    train_group_train.add_argument(
        "--total-steps",
        type=int,
        default=train_defaults.total_steps,
        help="总训练步数(默认: %(default)s)",
    )
    train_group_train.add_argument(
        "--grad-clip",
        type=float,
        default=train_defaults.grad_clip,
        help="梯度裁剪阈值(默认: %(default)s)",
    )
    train_group_train.add_argument(
        "--seed",
        type=int,
        default=train_defaults.seed,
        help="随机种子(默认: %(default)s)",
    )
    train_group_train.add_argument(
        "--num-workers",
        type=int,
        default=train_defaults.num_workers,
        help="DataLoader 线程数(默认: %(default)s)",
    )

    # 优化器
    train_group_opt = train_parser.add_argument_group("优化器配置")
    train_group_opt.add_argument(
        "--optimizer",
        type=str,
        default=train_defaults.optimizer_type,
        choices=["adam", "adamw", "sgd"],
        help="优化器类型(默认: %(default)s)",
    )
    train_group_opt.add_argument(
        "--lr",
        type=float,
        default=train_defaults.learning_rate,
        help="学习率(默认: %(default)s)",
    )
    train_group_opt.add_argument(
        "--weight-decay",
        type=float,
        default=train_defaults.weight_decay,
        help="权重衰减(默认: %(default)s)",
    )

    # 学习率调度
    train_group_sch = train_parser.add_argument_group("学习率调度配置")
    train_group_sch.add_argument(
        "--scheduler",
        type=str,
        default=train_defaults.scheduler_type,
        choices=["cosine_warmup", "cosine", "step", "exponential", "constant"],
        help="调度器类型(默认: %(default)s)",
    )
    train_group_sch.add_argument(
        "--warmup-steps",
        type=int,
        default=train_defaults.warmup_steps,
        help="预热步数(默认: %(default)s)",
    )
    train_group_sch.add_argument(
        "--min-lr-ratio",
        type=float,
        default=train_defaults.min_lr_ratio,
        help="最低学习率比例(默认: %(default)s)",
    )

    # 早停
    train_group_es = train_parser.add_argument_group("早停配置")
    train_group_es.add_argument(
        "--patience",
        type=int,
        default=train_defaults.early_stopping_patience,
        help="早停容忍轮数(默认: %(default)s)",
    )

    # 恢复训练
    train_parser.add_argument(
        "--resume", type=str, default=None, help="从检查点恢复训练(路径)"
    )

    # ============================================================
    # eval 子命令
    # ============================================================
    eval_parser = subparsers.add_parser("eval", help="评估模型")

    eval_parser.add_argument(
        "--checkpoint", type=str, required=True, help="模型检查点路径(必需)"
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="评估数据集(默认: %(default)s)",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=train_defaults.batch_size,
        help="评估批次大小(默认: %(default)s)",
    )

    # ============================================================
    # generate 子命令
    # ============================================================
    generate_parser = subparsers.add_parser("generate", help="生成文本")

    generate_parser.add_argument(
        "--checkpoint", type=str, required=True, help="模型检查点路径(必需)"
    )
    generate_parser.add_argument(
        "--prompt", type=str, default=None, help="输入提示文本,不指定则进入交互模式"
    )

    gen_group = generate_parser.add_argument_group("生成参数")
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=gen_defaults.max_new_tokens,
        help="最大生成 token 数(默认: %(default)s)",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=gen_defaults.temperature,
        help="温度,越高越随机(默认: %(default)s)",
    )
    gen_group.add_argument(
        "--top-k",
        type=int,
        default=gen_defaults.top_k,
        help="Top-K 采样(默认: %(default)s)",
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        default=gen_defaults.top_p,
        help="Top-P 核采样(默认: %(default)s)",
    )
    gen_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=gen_defaults.repetition_penalty,
        help="重复惩罚,>1.0 抑制重复(默认: %(default)s)",
    )
    gen_group.add_argument(
        "--no-kv-cache", action="store_true", default=False, help="禁用 KV 缓存"
    )

    return parser
