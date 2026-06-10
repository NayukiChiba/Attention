"""
src/data/dataset.py
创建Dataset和DataLoader
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config import GPTConfig, TrainingConfig, paths
from src.data.tokenizer import CharTokenizer


class NewsDataset(Dataset):
    """
    新闻文本数据集

    将长文本分割成固定长度的输入序列和目标序列, 以供GPT模型训练使用
    每一个块返回(input, target)对
    target是输入文本的右移版本, 以实现语言模型的自回归训练
        例如:
        输入文本: "这是一个测试文本。"
        输入序列: "<BOS>这是一个测试文本"
        目标序列: "这是一个测试文本<EOS>"
        这样模型在训练时会学习预测下一个字符, 从而实现语言建模


    """

    def __init__(
        self,
        file_path: Path,
        tokenizer: CharTokenizer,
        block_size: int,
    ):
        """

        Args:
            file_path (Path): 预处理后的数据文件路径
            tokenizer (CharTokenizer): 字符级分词器实例
            block_size (int): 输入序列的固定长度


        """
        self.tokenizer = tokenizer
        self.block_size = block_size

        print(f"加载数据集: {file_path} ...")

        texts = []

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                # 每行格式: "label\ttext"
                parts = line.strip().split("\t")
                # 只保留文本部分
                if len(parts) == 2:
                    texts.append(parts[1])

        print(f"数据集加载完成, 共 {len(texts)} 条文本")

        # 只保存文本，不预先编码（避免内存爆炸）
        self.texts = [t for t in texts if len(t) >= block_size]  # 过滤太短的文本
        print(f"过滤后剩余 {len(self.texts)} 条文本")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回第 idx 个样本的输入序列和目标序列（实时编码）

        处理逻辑:
        1. 取出第 idx 篇新闻文本
        2. 实时编码为 token ids
        3. 如果文本过长 (> block_size+1): 随机截取一段，增加训练多样性
        4. 如果文本过短 (< block_size+1): 用 <PAD> 填充到固定长度
        5. 切分为 input 和 target: input = ids[:-1], target = ids[1:] (右移)

        示例 (block_size=4):
            token_ids = [10, 20, 30, 40, 50]  (共5个token)
            input_ids  = [10, 20, 30, 40]     (前4个)
            target_ids = [20, 30, 40, 50]     (后4个，相当于input右移1位)

        这样模型学习: 看到 [10, 20, 30] 预测 30, 看到 [10, 20, 30, 40] 预测 50

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_ids, target_ids)
        """
        # 1. 取出原始文本
        text = self.texts[idx]

        # 2. 实时编码为 token ids
        token_ids = self.tokenizer.encode(text)

        # 3. 如果文本过长，随机截取一段 (block_size + 1 个token)
        if len(token_ids) > self.block_size + 1:
            import random

            max_start = len(token_ids) - self.block_size - 1
            start = random.randint(0, max_start)
            token_ids = token_ids[start : start + self.block_size + 1]

        # 4. 如果不够长，用 PAD 填充到 block_size + 1
        if len(token_ids) < self.block_size + 1:
            pad_id = self.tokenizer.pad_token_id
            token_ids = token_ids + [pad_id] * (self.block_size + 1 - len(token_ids))

        # 5. 切分为输入和目标 (target 是 input 右移一位)
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)  # 前 block_size 个
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)  # 后 block_size 个
        return input_ids, target_ids


def create_dataloader(
    tokenizer: CharTokenizer,
    gpt_config: GPTConfig,
    training_config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练集、验证集和测试集的 DataLoader
    Args:
        tokenizer (CharTokenizer): 分词器实例
        gpt_config (GPTConfig): GPT 模型配置对象
        training_config (TrainingConfig): 训练配置对象
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (训练集, 验证集, 测试集)的 DataLoader
    """
    train_dataset = NewsDataset(
        paths.INTERIM_TRAIN_DATASET_PATH,
        tokenizer,
        block_size=gpt_config.context_length,
    )
    val_dataset = NewsDataset(
        paths.INTERIM_VAL_DATASET_PATH,
        tokenizer,
        block_size=gpt_config.context_length,
    )
    test_dataset = NewsDataset(
        paths.INTERIM_TEST_DATASET_PATH,
        tokenizer,
        block_size=gpt_config.context_length,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=training_config.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=training_config.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 创建分词器实例
    vocab_path = paths.PROCESSED_DATASETS_DIR / "vocab.json"
    tokenizer = CharTokenizer.load(vocab_path)

    # 创建 DataLoader
    gpt_config = GPTConfig(vocab_size=tokenizer.vocab_size)
    training_config = TrainingConfig(
        batch_size=4, num_workers=0
    )  # 小批量和单线程用于测试
    train_loader, val_loader, test_loader = create_dataloader(
        tokenizer, gpt_config, training_config
    )

    # 测试一个 batch
    print("测试训练集第一个 batch:")
    for input_ids, target_ids in train_loader:
        print(f"input_ids shape: {input_ids.shape}")  # [batch_size, block_size]
        print(f"target_ids shape: {target_ids.shape}")
        print(f"input_ids[0][:10]: {input_ids[0][:10].tolist()}")
        print(f"target_ids[0][:10]: {target_ids[0][:10].tolist()}")

        # 验证 target 是 input 右移 1 位
        assert torch.all(input_ids[:, 1:] == target_ids[:, :-1]), (
            "target 不是 input 右移！"
        )
        print("target 正确右移")
        break

    # 解码测试
    print("解码测试:")
    sample_input = input_ids[0][:50]  # 取前 50 个 token
    decoded_text = tokenizer.decode(sample_input.tolist())
    print(f"  解码文本: {decoded_text}")
