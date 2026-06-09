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


class GPTEmbedding(nn.Module):
    """
    GPT Embedding 层 = Token Embedding + Position Encoding + Dropout
    token变成向量 + 加上位置编码 + dropout

    将 token ids 转换为包含位置信息的嵌入向量

    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        dropout_rate: float = 0.1,
        pos_encoding_type: str = "learnable",
    ):
        """
        Args:
            vocab_size (int): 词表大小
            embedding_dim (int): 嵌入维度
            block_size (int): 上下文长度（最大 token 数）
            dropout_rate (float): Dropout 比例
            pos_encoding_type (str): 位置编码类型 ("sinusoidal" 或 "learnable")
        """

        super().__init__()
        self.embedding_dim = embedding_dim
        self.block_size = block_size

        # Token Embedding 层
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)

        # 位置编码层
        if pos_encoding_type == "sinusoidal":
            # 固定的正弦位置编码
            # 注册为 buffer, 不参与训练, 但会随模型保存和加载
            self.register_buffer(
                "pos_embedding",
                self._create_sinusoidal_positional_encoding(block_size, embedding_dim),
            )
        elif pos_encoding_type == "learnable":
            # 可学习的位置编码, GPT-2 风格
            self.pos_embedding = nn.Embedding(block_size, embedding_dim)
        else:
            raise ValueError(f"不支持的 pos_encoding_type: {pos_encoding_type}")

        self.pos_encoding_type = pos_encoding_type
        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

    def _create_sinusoidal_positional_encoding(
        self, max_seq_length: int, embedding_dim: int
    ) -> torch.Tensor:
        """
        生成正弦位置编码矩阵
        公式:
            position_embedding(pos, 2i) = sin(pos / 10000^(2i/embedding_dim))
            position_embedding(pos, 2i+1) = cos(pos / 10000^(2i/embedding_dim))
        Args:
            max_seq_length (int): 最大序列长度
            embedding_dim (int): 嵌入维度
        Returns:
            torch.Tensor: 形状为 (max_seq_length, embedding_dim) 的位置编码矩阵
        """
        # 初始化位置编码矩阵

        position_embedding = torch.zeros(max_seq_length, embedding_dim)
        # shape=(max_seq_length, embedding_dim//2), 每个位置的偶数维用 sin, 奇数维用 cos
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )  # shape=(embedding_dim//2,)

        position_embedding[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        position_embedding[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        return position_embedding  # shape=(max_seq_length, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): 输入的 token id 张量, 形状为 (batch_size, seq_length)

        Returns:
            embeddings:shape=(batch_size, seq_length, embedding_dim)
        """
        batch_size, seq_length = input_ids.shape
        # 1. 获取 token 嵌入
        token_embedding = self.token_embedding(
            input_ids
        )  # shape=(batch_size, seq_length, embedding_dim)

        # 2. 获取位置编码
        if self.pos_encoding_type == "learnable":
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(
                0
            )  # shape=(1, seq_length)
            position_embedding = self.pos_embedding(
                position_ids
            )  # shape=(1, seq_length, embedding_dim)
        else:  # sinusoidal
            position_embedding = self.pos_embedding[:seq_length, :].unsqueeze(
                0
            )  # shape=(1, seq_length, embedding_dim)

        # 3. 将 token 嵌入和位置编码相加
        embeddings = (
            token_embedding + position_embedding
        )  # shape=(batch_size, seq_length, embedding_dim)

        # 4. 应用 dropout
        embeddings = self.dropout(
            embeddings
        )  # shape=(batch_size, seq_length, embedding_dim)

        # 应用 dropout
        return embeddings  # shape=(batch_size, seq_length, embedding_dim)


if __name__ == "__main__":
    from config import GPTConfig

    config = GPTConfig()
    embedding_layer = GPTEmbedding(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        block_size=config.context_length,
        dropout_rate=config.dropout_rate,
        pos_encoding_type=config.pos_encoding_type,
    )

    embeddings_sinusoidal = GPTEmbedding(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        block_size=config.context_length,
        dropout_rate=config.dropout_rate,
        pos_encoding_type="sinusoidal",
    )

    # 测试输入
    batch_size = 4
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # 前向传播测试
    output = embedding_layer(input_ids)
    print(
        "输出形状 (learnable):", output.shape
    )  # 应该是 (batch_size, seq_length, embedding_dim)

    output_sinusoidal = embeddings_sinusoidal(input_ids)
    print("输出形状 (sinusoidal):", output_sinusoidal.shape)

    # 测试位置编码是否正确叠加
    # 验证形状
    assert output.shape == (batch_size, seq_length, config.embedding_dim)
    assert output_sinusoidal.shape == (batch_size, seq_length, config.embedding_dim)
    print("✅ 形状测试通过")

    # 参数量统计
    print(
        f"\n可学习位置编码参数量: {sum(p.numel() for p in embedding_layer.parameters()):,}"
    )
    print(
        f"正弦位置编码参数量: {sum(p.numel() for p in embeddings_sinusoidal.parameters()):,}"
    )
