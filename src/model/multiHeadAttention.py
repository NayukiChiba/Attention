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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算多头注意力输出
        Args:
            query: Query矩阵 (batch_size, seq_length_q, embedding_dim)
            key: Key矩阵 (batch_size, seq_length_k, embedding_dim)
            value: Value矩阵 (batch_size, seq_length_v, embedding_dim)
            mask: 可选的mask矩阵 (batch_size, seq_length_q, seq_length_k) 或者 (batch_size, 1, seq_length_k)
            return_attention_weights: 是否返回注意力权重
        Returns:
            if return_attention_weights:
                output: 注意力输出 (batch_size, seq_length_q, embedding_dim)
                attention_weights: 注意力权重 (batch_size, num_attention_heads, seq_length_q, seq_length_k)
            else:
                output: 注意力输出 (batch_size, seq_length_q, embedding_dim)
        """
        # step1: 线性投影
        Q = self.W_Q(query)  # (batch_size, seq_length_q, embedding_dim)
        K = self.W_K(key)  # (batch_size, seq_length_k, embedding_dim)
        V = self.W_V(value)  # (batch_size, seq_length_v, embedding_dim)

        # step2: 拆分多头
        # (batch_size, seq_length, embedding_dim) -> (batch_size, num_attention_heads, seq_length, head_dim)
        Q = self.split_heads(
            Q
        )  # (batch_size, num_attention_heads, seq_length_q, head_dim)
        K = self.split_heads(
            K
        )  # (batch_size, num_attention_heads, seq_length_k, head_dim)
        V = self.split_heads(
            V
        )  # (batch_size, num_attention_heads, seq_length_v, head_dim)

        # step3: 并行计算注意力
        # output: (batch_size, num_attention_heads, seq_length_q, head_dim)
        # attention_weights: (batch_size, num_attention_heads, seq_length_q, seq_length_k)
        if return_attention_weights:
            attention_output, attention_weights = self.scaled_dot_product_attention(
                Q,
                K,
                V,
                mask=mask,
                return_attention_weights=True,
            )
        else:
            attention_output = self.scaled_dot_product_attention(
                Q,
                K,
                V,
                mask=mask,
                return_attention_weights=False,
            )

        # step4: 拼接结果
        # (batch_size, num_attention_heads, seq_length_q, head_dim) -> (batch_size, seq_length_q, embedding_dim)
        attention_output = self.combine_heads(attention_output)

        # step5: 输入投影
        # (batch_size, seq_length_q, embedding_dim) -> (batch_size, seq_length_q, embedding_dim)
        output = self.W_O(attention_output)

        # step6: dropout
        output = self.dropout(output)

        # 返回结果
        if return_attention_weights:
            return output, attention_weights
        else:
            return output


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Multi-Head Attention")
    print("=" * 60)

    # 设置参数
    batch_size = 2
    seq_len = 10
    embedding_dim = 512
    num_attention_heads = 8
    dropout_rate = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}\n")

    # 创建模型
    mha = MultiHeadAttention(
        embedding_dim=embedding_dim,
        num_attention_heads=num_attention_heads,
        dropout_rate=dropout_rate,
    ).to(device)

    print("模型参数:")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  num_attention_heads: {num_attention_heads}")
    print(f"  head_dim: {embedding_dim // num_attention_heads}")
    print(f"  总参数量: {sum(p.numel() for p in mha.parameters()):,}")
    print()

    # 1. 测试自注意力（Self-Attention）
    print("1. 自注意力（Self-Attention）")
    print("-" * 60)
    # 创建输入（Q = K = V）
    x = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    print(f"Input shape: {x.shape}")

    # 前向传播
    output = mha(x, x, x)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, embedding_dim), "输出形状错误"
    print("✓ 自注意力测试通过\n")

    # 2. 测试带 mask 的自注意力
    print("2. 带 mask 的自注意力")
    print("-" * 60)

    from src.model.mask import create_causal_mask

    # 创建因果掩码
    causal_mask = create_causal_mask(seq_len, device=device)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    print(f"Causal mask shape: {causal_mask.shape}")

    # 前向传播
    output_with_mask = mha(x, x, x, mask=causal_mask)
    print(f"Output with mask shape: {output_with_mask.shape}")
    assert output_with_mask.shape == (batch_size, seq_len, embedding_dim), (
        "带mask的输出形状错误"
    )
    print("✓ 带 mask 的自注意力测试通过\n")

    # 3. 测试返回注意力权重
    print("3. 返回注意力权重")
    print("-" * 60)

    mha.eval()  # 关闭 dropout
    with torch.no_grad():
        output, attn_weights = mha(
            x, x, x, mask=causal_mask, return_attention_weights=True
        )

    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    assert attn_weights.shape == (batch_size, num_attention_heads, seq_len, seq_len), (
        "注意力权重形状错误"
    )

    # 验证注意力权重
    row_sums = attn_weights.sum(dim=-1)
    print(f"Attention weights 每行求和（前5个）: {row_sums[0, 0, :5]}")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), (
        "注意力权重每行求和不为1"
    )
    print("✓ 注意力权重测试通过\n")

    # 4. 测试 Cross-Attention（不同序列长度）
    print("4. Cross-Attention（不同序列长度）")
    print("-" * 60)

    seq_len_q = 8
    seq_len_kv = 12

    query = torch.randn(batch_size, seq_len_q, embedding_dim, device=device)
    key = torch.randn(batch_size, seq_len_kv, embedding_dim, device=device)
    value = torch.randn(batch_size, seq_len_kv, embedding_dim, device=device)

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")

    output_cross = mha(query, key, value)
    print(f"Output shape: {output_cross.shape}")
    assert output_cross.shape == (batch_size, seq_len_q, embedding_dim), (
        "Cross-attention 输出形状错误"
    )
    print("✓ Cross-Attention 测试通过\n")

    # 5. 测试 split_heads 和 combine_heads
    print("5. 测试 split_heads 和 combine_heads")
    print("-" * 60)

    test_tensor = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    print(f"原始张量 shape: {test_tensor.shape}")

    # 拆分
    split_tensor = mha.split_heads(test_tensor)
    print(f"拆分后 shape: {split_tensor.shape}")
    assert split_tensor.shape == (
        batch_size,
        num_attention_heads,
        seq_len,
        embedding_dim // num_attention_heads,
    ), "拆分形状错误"

    # 合并
    combined_tensor = mha.combine_heads(split_tensor)
    print(f"合并后 shape: {combined_tensor.shape}")
    assert combined_tensor.shape == (batch_size, seq_len, embedding_dim), "合并形状错误"

    # 验证拆分-合并是可逆的
    assert torch.allclose(test_tensor, combined_tensor), "拆分-合并不可逆"
    print("✓ split_heads 和 combine_heads 测试通过\n")

    # 6. 测试梯度反向传播
    print("6. 测试梯度反向传播")
    print("-" * 60)

    mha.train()
    x_grad = torch.randn(
        batch_size, seq_len, embedding_dim, device=device, requires_grad=True
    )

    output_grad = mha(x_grad, x_grad, x_grad)
    loss = output_grad.sum()
    loss.backward()

    print(f"Input 梯度存在: {x_grad.grad is not None}")
    print(f"W_Q 权重梯度存在: {mha.W_Q.weight.grad is not None}")
    print(f"W_O 权重梯度存在: {mha.W_O.weight.grad is not None}")
    assert x_grad.grad is not None, "输入梯度不存在"
    assert mha.W_Q.weight.grad is not None, "W_Q 梯度不存在"
    print("✓ 梯度反向传播测试通过\n")

    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
