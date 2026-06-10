"""
src/cli/menu.py

交互式菜单
"""

from pathlib import Path

import torch

from config import paths
from config.defaults import DataConfig, GenerationConfig, GPTConfig, TrainingConfig


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


def _menu_train():
    """交互式训练"""
    # 1. 加载分词器
    print("\n加载分词器...")
    tokenizer = _load_tokenizer()
    print(f"词表大小: {tokenizer.vocab_size}")

    # 2. 创建配置
    model_config = GPTConfig(vocab_size=tokenizer.vocab_size)
    training_config = TrainingConfig()

    # 3. 询问是否修改默认参数
    print("\n当前默认参数:")
    print(f"  批次大小: {training_config.batch_size}")
    print(f"  训练轮数: {training_config.max_epochs}")
    print(f"  学习率: {training_config.learning_rate}")
    print(f"  优化器: {training_config.optimizer_type}")

    custom = input("\n是否修改参数？(y/n，默认 n): ").strip().lower()
    if custom == "y":
        try:
            bs = input(f"批次大小 ({training_config.batch_size}): ").strip()
            if bs:
                training_config.batch_size = int(bs)

            ep = input(f"训练轮数 ({training_config.max_epochs}): ").strip()
            if ep:
                training_config.max_epochs = int(ep)

            lr = input(f"学习率 ({training_config.learning_rate}): ").strip()
            if lr:
                training_config.learning_rate = float(lr)
        except ValueError as e:
            print(f"参数格式错误: {e}，使用默认参数")

    # 4. 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader, _ = _create_dataloaders(
        tokenizer, model_config, training_config
    )

    # 5. 创建模型
    print("创建模型...")
    from src.model.gpt import GPT

    model = GPT(model_config)

    # 6. 询问是否从检查点恢复
    resume_from = None
    if paths.LAST_MODEL_PATH.exists():
        resume = (
            input("\n发现已有检查点，是否恢复训练？(y/n，默认 n): ").strip().lower()
        )
        if resume == "y":
            resume_from = paths.LAST_MODEL_PATH

    # 7. 创建训练器并开始训练
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

    trainer.train(resume_from=resume_from)


def _menu_eval():
    """交互式评估"""
    from src.evaluate.evaluator import Evaluator
    from src.model.gpt import GPT

    # 1. 选择检查点
    checkpoint_path = _select_checkpoint()
    if checkpoint_path is None:
        return

    print(f"\n加载检查点: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 2. 加载分词器
    print("加载分词器...")
    tokenizer = _load_tokenizer()

    # 3. 获取模型配置
    if "config" in checkpoint:
        model_config = checkpoint["config"]
    else:
        model_config = GPTConfig(vocab_size=tokenizer.vocab_size)

    # 4. 创建模型并加载权重
    print("创建模型...")
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 5. 选择评估数据集
    print("\n选择评估数据集:")
    print("  1. 测试集")
    print("  2. 验证集")
    print("  3. 训练集")

    split_choice = input("请选择 (1-3, 默认 1): ").strip()
    split_map = {"1": "test", "2": "val", "3": "train", "": "test"}
    split = split_map.get(split_choice, "test")

    # 6. 创建数据加载器
    training_config = TrainingConfig()
    train_loader, val_loader, test_loader = _create_dataloaders(
        tokenizer, model_config, training_config
    )

    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    eval_loader = loader_map[split]

    # 7. 评估
    evaluator = Evaluator(model, eval_loader, training_config)
    metrics = evaluator.evaluate()

    print()
    print("=" * 60)
    print(f"评估结果 ({split} 集)")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 60)


def _menu_generate():
    """交互式生成"""
    from src.inference.generator import TextGenerator
    from src.model.gpt import GPT

    # 1. 选择检查点
    checkpoint_path = _select_checkpoint()
    if checkpoint_path is None:
        return

    print(f"\n加载检查点: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 2. 加载分词器
    print("加载分词器...")
    tokenizer = _load_tokenizer()

    # 3. 获取模型配置
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
    gen_config = GenerationConfig()
    generator = TextGenerator(model, tokenizer, gen_config, device=device)

    # 6. 交互生成循环
    print()
    print("进入交互生成模式（输入 'quit' 退出）")
    print(
        f"生成参数: temperature={gen_config.temperature}, top_k={gen_config.top_k}, top_p={gen_config.top_p}"
    )
    print("-" * 60)

    while True:
        prompt = input("\n请输入提示文本: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("退出生成模式")
            break
        if not prompt:
            print("提示文本不能为空")
            continue

        print("\n生成中...")
        result = generator.generate(prompt)
        print()
        print("生成结果:")
        print(result)
        print("-" * 60)


def _select_checkpoint() -> Path | None:
    """选择检查点文件"""
    checkpoints = sorted(
        paths.CHECKPOINTS_DIR.glob("*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not checkpoints:
        print("\n没有找到检查点文件，请先训练模型！")
        return None

    print("\n可用的检查点:")
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"  {i}. {ckpt.name}")

    choice = input(f"请选择 (1-{len(checkpoints)}, 默认 1): ").strip()
    if not choice:
        choice = "1"

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(checkpoints):
            return checkpoints[idx]
    except ValueError:
        pass

    print("选择无效，使用第一个检查点")
    return checkpoints[0]


def print_banner():
    """打印欢迎 Banner"""
    banner = r"""
  ██████╗ ██████╗ ████████╗
 ██╔════╝ ██╔══██╗╚══██╔══╝
 ██║  ███╗██████╔╝   ██║
 ██║   ██║██╔═══╝    ██║
 ╚██████╔╝██║        ██║
  ╚═════╝ ╚═╝        ╚═╝
 中文新闻文本生成模型
 """
    print(banner)


def show_menu():
    """显示并处理交互式菜单"""
    print_banner()

    while True:
        print("\n主菜单:")
        print("  1. 训练模型")
        print("  2. 评估模型")
        print("  3. 生成文本")
        print("  0. 退出")

        choice = input("\n请选择操作 (0-3): ").strip()

        if choice == "0":
            print("\n退出程序，再见！")
            break
        elif choice == "1":
            _menu_train()
        elif choice == "2":
            _menu_eval()
        elif choice == "3":
            _menu_generate()
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    show_menu()
