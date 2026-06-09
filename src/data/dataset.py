"""
src/data/dataset.py
创建Dataset和DataLoader
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

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
        max_samples: int | None = None,
    ):
        """

        Args:
            file_path (Path): 预处理后的数据文件路径
            tokenizer (CharTokenizer): 字符级分词器实例
            block_size (int): 输入序列的固定长度
            max_samples (int|None): 最大样本数量, None表示使用全部数据


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

        # 编码文本并创建输入-目标对
        print("正在编码文本")
        all_token_ids = []
        for text in texts:
            token_ids = tokenizer.encode(text)
            all_token_ids.extend(token_ids)

        print(f"总 token 数: {len(all_token_ids)}")

        # 切分为固定长度的块
        # 每个块需要 block_size + 1 个 token 来创建(input, target)对
        # 例如 block_size=4, 输入序列需要 4 个 token, 目标序列需要 4 个 token, 共 5 个 token 来创建一个样本
        self.samples = []
        for i in range(0, len(all_token_ids) - block_size, block_size):
            # 切分出一个块, 包含 block_size + 1 个 token
            chunk = all_token_ids[i : i + block_size + 1]
            if len(chunk) == block_size + 1:
                # chunk[:-1] 是输入序列, chunk[1:] 是目标序列
                self.samples.append(chunk)
            # 如果设置了 max_samples, 则在达到最大样本数量后停止切分
            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"数据集切分完成, 共 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回第 idx 个样本的输入序列和目标序列
        输入序列是 chunk[:-1], 目标序列是 chunk[1:]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (input_ids, target_ids)
        """
        chunk = self.samples[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)  # 输入序列
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)  # 目标序列
        return input_ids, target_ids
