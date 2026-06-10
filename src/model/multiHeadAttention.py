"""
src/model/multiHeadAttention.py

多头注意力机制的实现(Multi-Head Attention)

"""

import torch
import torch.nn as nn

from src.model.scaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    将输入投影到多个子空间中, 在每一个子空间中独立计算注意力, 最后拼接结果

    计算流程:
        1. 线性投影: Q, K, V = X @ W_Q, X @ W_K, X @ W_V
        2. 拆分多头: 将embedding_dim 拆分成 num_attention_heads * head_dim
        3. 并行计算: 在每一个头上独立计算注意力
        4. 拼接结果: 将每个头的输出拼接成 embedding_dim 维度
        5. 输入投影: output = concat(head_1, head_2, ..., head_n) @ W_O

    Args:
        embedding_dim: 输入和输出的维度
        num_attention_heads: 注意力头的数量
        dropout_rate: 注意力权重的dropout概率


    """

    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        dropout_rate: float = 0.1,
    ):
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % num_attention_heads == 0, (
            "embedding_dim must be divisible by num_attention_heads"
        )
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = embedding_dim // num_attention_heads

        # 线性投影层
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)
        self.W_O = nn.Linear(embedding_dim, embedding_dim)

        # 缩放点积注意力机制
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_rate)

        # dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入张量拆分成多个头
        Args:
            x: 输入张量 (batch_size, seq_length, embedding_dim)
        Returns:
            拆分后的张量 (batch_size, num_attention_heads, seq_length, head_dim)
        """
        batch_size, seq_length, embedding_dim = x.size()
        # shape: (batch_size, seq_length, num_attention_heads, head_dim)
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        # 调整维度顺序为 (batch_size, num_attention_heads, seq_length, head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多个头的输出拼接成一个张量
        Args:
            x: 输入张量 (batch_size, num_attention_heads, seq_length, head_dim)
        Returns:
            拼接后的张量 (batch_size, seq_length, embedding_dim)
        """
        batch_size, num_attention_heads, seq_length, head_dim = x.size()
        # 调整维度顺序为 (batch_size, seq_length, num_attention_heads, head_dim)
        x = x.transpose(1, 2)
        # 调整为 (batch_size, seq_length, embedding_dim)
        x = x.contiguous().view(batch_size, seq_length, self.embedding_dim)

        return x
