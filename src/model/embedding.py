"""
src/model/embedding.py
GPT模型的嵌入层实现
词嵌入层 + 位置嵌入层
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token Embedding 层

    将 token id 映射为稠密向量

    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Args:
            vocab_size (int): 词表大小
            embedding_dim (int): 嵌入维度

        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): 输入的 token id 张量, 形状为 (batch_size, seq_length)

        Returns:
            embeddings:shape=(batch_size, seq_length, embedding_dim)
        """
        # 标准做法: 乘以 sqrt(embedding_dim),放大嵌入向量
        # 原因: 位置编码的数值尺度匹配(位置编码用 sin/cos,范围在 [-1, 1])
        return self.embedding(input_ids) * torch.sqrt(
            torch.tensor(self.embedding_dim, dtype=torch.float32)
        )
