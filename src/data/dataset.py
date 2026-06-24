"""
src/data/dataset.py
创建Dataset和DataLoader
"""

import random
import re
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
        # 不按 block_size 过滤短文本，保留完整短句有助于模型学习句子结束。
        self.texts = [t for t in texts if len(t) >= 4]
        print(f"过滤后剩余 {len(self.texts)} 条文本")

    def __len__(self):
        return len(self.texts)

    def _splitIntoSentences(self, text: str) -> list[str]:
        """按中文标点切分句子，保留句末标点。"""
        sentences = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _selectTextSpan(self, text: str) -> str:
        """优先选择完整句段，过长时再退化为连续片段。"""
        maxTokenCount = self.block_size + 1
        minTokenCount = max(8, min(self.block_size // 8, 24))
        candidates = []

        sentences = self._splitIntoSentences(text)
        for startIndex in range(len(sentences)):
            spanParts = []
            for sentence in sentences[startIndex:]:
                spanParts.append(sentence)
                spanText = "".join(spanParts)
                tokenCount = len(self.tokenizer.encode(spanText))
                if tokenCount > maxTokenCount:
                    break
                if tokenCount >= minTokenCount:
                    candidates.append(spanText)

        if candidates:
            return random.choice(candidates)

        token_ids = self.tokenizer.encode(text)
        if len(token_ids) <= maxTokenCount:
            return text

        # 找不到合适完整句时，随机取一段固定长度上下文。
        start = random.randint(0, len(token_ids) - maxTokenCount)
        spanIds = token_ids[start : start + maxTokenCount]
        return self.tokenizer.decode(spanIds)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回第 idx 个样本的输入序列和目标序列（实时编码）

        优先使用完整句段训练，让模型学习自然起止和句末标点。
        """

        # 1. 取出原始文本
        text = self.texts[idx]

        # 2. 实时编码为 token ids，保留 BOS/EOS。
        spanText = self._selectTextSpan(text)
        token_ids = self.tokenizer.encode(spanText)
        token_ids = token_ids[: self.block_size + 1]

        # 3. 切分为输入和目标 (target 是 input 右移一位)
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_ids, target_ids


def collateBatch(batch: list, pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """将不等长序列 padding 到相同长度"""
    inputs, targets = zip(*batch)
    maxLen = max(x.shape[0] for x in inputs)

    paddedInputs = []
    paddedTargets = []
    for inp, tgt in zip(inputs, targets):
        curLen = inp.shape[0]
        if curLen < maxLen:
            pad = torch.full((maxLen - curLen,), pad_id, dtype=torch.long)
            paddedInputs.append(torch.cat([inp, pad]))
            paddedTargets.append(torch.cat([tgt, pad]))
        else:
            paddedInputs.append(inp)
            paddedTargets.append(tgt)

    return torch.stack(paddedInputs), torch.stack(paddedTargets)


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

    # 判断设备类型，CUDA 时启用 pin_memory 加速 CPU -> GPU 传输
    device_type = training_config.device.split(":")[0]
    pin_memory = device_type == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=training_config.num_workers > 0,
        collate_fn=collateBatch,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=training_config.num_workers > 0,
        collate_fn=collateBatch,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=training_config.num_workers > 0,
        collate_fn=collateBatch,
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

        # 验证每条样本的有效 token 区间是右移关系。
        # padding 后不能直接整批比较，否则 EOS 和 PAD 边界会误报。
        padId = tokenizer.pad_token_id
        for rowIndex in range(input_ids.shape[0]):
            validLength = int((input_ids[rowIndex] != padId).sum().item())
            if validLength <= 1:
                continue

            assert torch.all(
                input_ids[rowIndex, 1:validLength]
                == target_ids[rowIndex, : validLength - 1]
            ), f"第 {rowIndex} 条样本 target 不是 input 右移！"

        print("target 正确右移")
        break

    # 解码测试
    print("解码测试:")
    sample_input = input_ids[0][:50]  # 取前 50 个 token
    decoded_text = tokenizer.decode(sample_input.tolist())
    print(f"  解码文本: {decoded_text}")
