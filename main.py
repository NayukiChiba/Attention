"""
main.py

主程序入口

用法:
    python main.py train [options]
    python main.py eval [options]
    python main.py generate [options]
    python main.py               (交互式菜单)
"""

import sys
from pathlib import Path

import torch

from config import paths
from config.defaults import DataConfig, GenerationConfig, GPTConfig, TrainingConfig
from src.cli.menu import show_menu
from src.cli.parser import create_parser


def _build_tokenizer(data_config: DataConfig = None):
    """从训练数据构建分词器"""
    from src.data.tokenizer import CharTokenizer

    data_config = data_config or DataConfig()
    tokenizer = CharTokenizer(data_config)

    # 读取训练集文本
    texts = []
    with paths.INTERIM_TRAIN_DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                texts.append(parts[1])

    # 构建词表
    tokenizer.build_vocab(texts)

    # 保存词表
    vocab_path = paths.PROCESSED_DATASETS_DIR / "vocab.json"
    tokenizer.save(vocab_path)

    return tokenizer


def _load_tokenizer():
    """加载已保存的分词器"""
    from src.data.tokenizer import CharTokenizer

    vocab_path = paths.PROCESSED_DATASETS_DIR / "vocab.json"
    if vocab_path.exists():
        return CharTokenizer.load(vocab_path)

    # 词表不存在，需要先构建
    print("词表文件不存在，开始构建...")
    return _build_tokenizer()


def _create_dataloaders(tokenizer, gpt_config, training_config):
    """创建数据加载器"""
    from src.data.dataset import create_dataloader

    return create_dataloader(tokenizer, gpt_config, training_config)


def _build_model_config(args, tokenizer) -> GPTConfig:
    """从命令行参数构建模型配置"""
    config = GPTConfig()

    # 用命令行参数覆盖默认值
    arg_map = {
        "vocab_size": getattr(args, "vocab_size", None),
        "context_length": getattr(args, "context_length", None),
        "embedding_dim": getattr(args, "embedding_dim", None),
        "num_attention_heads": getattr(args, "num_heads", None),
        "num_layers": getattr(args, "num_layers", None),
        "ffn_hidden_dim": getattr(args, "ffn_hidden_dim", None),
        "dropout_rate": getattr(args, "dropout", None),
        "pos_encoding_type": getattr(args, "pos_encoding", None),
        "activation": getattr(args, "activation", None),
        "norm_type": getattr(args, "norm_type", None),
    }

    for field_name, arg_value in arg_map.items():
        if arg_value is not None:
            setattr(config, field_name, arg_value)

    # 词表大小以实际 tokenizer 为准
    config.vocab_size = tokenizer.vocab_size

    return config


def _build_training_config(args) -> TrainingConfig:
    """从命令行参数构建训练配置"""
    config = TrainingConfig()

    arg_map = {
        "batch_size": getattr(args, "batch_size", None),
        "max_epochs": getattr(args, "epochs", None),
        "total_steps": getattr(args, "total_steps", None),
        "grad_clip": getattr(args, "grad_clip", None),
        "seed": getattr(args, "seed", None),
        "num_workers": getattr(args, "num_workers", None),
        "optimizer_type": getattr(args, "optimizer", None),
        "learning_rate": getattr(args, "lr", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "scheduler_type": getattr(args, "scheduler", None),
        "warmup_steps": getattr(args, "warmup_steps", None),
        "min_lr_ratio": getattr(args, "min_lr_ratio", None),
        "early_stopping_patience": getattr(args, "patience", None),
    }

    for field_name, arg_value in arg_map.items():
        if arg_value is not None:
            setattr(config, field_name, arg_value)

    return config


def _build_generation_config(args) -> GenerationConfig:
    """从命令行参数构建生成配置"""
    config = GenerationConfig()

    arg_map = {
        "max_new_tokens": getattr(args, "max_tokens", None),
        "temperature": getattr(args, "temperature", None),
        "top_k": getattr(args, "top_k", None),
        "top_p": getattr(args, "top_p", None),
        "repetition_penalty": getattr(args, "repetition_penalty", None),
    }

    for field_name, arg_value in arg_map.items():
        if arg_value is not None:
            setattr(config, field_name, arg_value)

    if getattr(args, "no_kv_cache", False):
        config.use_kv_cache = False

    return config


def train_main(args):
    """训练入口"""
    # 1. 加载或构建分词器
    print("=" * 60)
    print("准备训练")
    print("=" * 60)
    print()

    print("加载分词器...")
    tokenizer = _load_tokenizer()
    print(f"词表大小: {tokenizer.vocab_size}")

    # 2. 构建配置
    model_config = _build_model_config(args, tokenizer)
    training_config = _build_training_config(args)

    print()
    print("模型配置:")
    model_config._summary()
    print()
    print("训练配置:")
    training_config._summary()
    print()

    # 3. 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, _ = _create_dataloaders(
        tokenizer, model_config, training_config
    )

    # 4. 创建模型
    print("创建模型...")
    from src.model.gpt import GPT

    model = GPT(model_config)

    # 5. 创建训练器
    from src.train.trainer import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        checkpoint_dir=paths.CHECKPOINTS_DIR,
        log_dir=paths.LOGS_DIR,
        tensorboard_dir=paths.TENSORBOARD_DIR,
    )

    # 6. 开始训练
    resume_from = None
    if getattr(args, "resume", None):
        resume_from = Path(args.resume)

    trainer.train(resume_from=resume_from)


def eval_main(args):
    """评估入口"""
    from src.evaluate.evaluator import Evaluator
    from src.model.gpt import GPT

    # 1. 加载检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"检查点文件不存在: {checkpoint_path}")
        sys.exit(1)

    print("加载检查点:", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. 加载分词器
    print("加载分词器...")
    tokenizer = _load_tokenizer()

    # 3. 从检查点获取模型配置
    if "config" in checkpoint:
        model_config = checkpoint["config"]
    else:
        model_config = GPTConfig(vocab_size=tokenizer.vocab_size)

    print(f"词表大小: {tokenizer.vocab_size}")

    # 4. 创建模型并加载权重
    print("创建模型...")
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 5. 创建数据加载器
    training_config = _build_training_config(args)
    _, _, test_loader = _create_dataloaders(tokenizer, model_config, training_config)

    # 如果指定了 split，用对应的数据加载器
    if hasattr(args, "split"):
        if args.split == "train":
            test_loader, _, _ = _create_dataloaders(
                tokenizer, model_config, training_config
            )
        elif args.split == "val":
            _, test_loader, _ = _create_dataloaders(
                tokenizer, model_config, training_config
            )

    # 6. 评估
    evaluator = Evaluator(model, test_loader, training_config)
    metrics = evaluator.evaluate()

    print()
    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 60)


def generate_main(args):
    """生成入口"""
    from src.inference.generator import TextGenerator
    from src.model.gpt import GPT

    # 1. 加载检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"检查点文件不存在: {checkpoint_path}")
        sys.exit(1)

    print("加载检查点:", checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 2. 加载分词器
    print("加载分词器...")
    tokenizer = _load_tokenizer()

    # 3. 从检查点获取模型配置
    if "config" in checkpoint:
        model_config = checkpoint["config"]
    else:
        model_config = GPTConfig(vocab_size=tokenizer.vocab_size)

    # 4. 创建模型并加载权重
    print("创建模型...")
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 5. 创建生成器
    gen_config = _build_generation_config(args)
    generator = TextGenerator(model, tokenizer, gen_config, device=device)

    # 6. 生成文本
    if args.prompt:
        # 单次生成
        print()
        print("=" * 60)
        print("提示:", args.prompt)
        print("-" * 60)
        result = generator.generate(args.prompt)
        print(result)
        print("=" * 60)
    else:
        # 交互模式
        print()
        print("进入交互生成模式（输入 'quit' 退出）")
        print("-" * 60)
        while True:
            prompt = input("\n请输入提示文本: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                print("退出生成模式")
                break
            if not prompt:
                continue

            result = generator.generate(prompt)
            print()
            print("生成结果:")
            print(result)


def main():
    """主函数"""
    # 如果没有命令行参数，启动交互式菜单
    if len(sys.argv) == 1:
        show_menu()
        return

    # 否则使用 CLI 模式
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_main(args)

    elif args.command == "eval":
        eval_main(args)

    elif args.command == "generate":
        generate_main(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
