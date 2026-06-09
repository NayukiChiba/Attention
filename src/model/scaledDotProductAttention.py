"""
src/model/scaledDotProductAttention.py

缩放点积注意力机制的实现(Scaled Dot-Product Attention)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制的实现
    计算公式:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    其中:
        Q: 查询矩阵 (Query)
        K: 键矩阵 (Key)
        V: 值矩阵 (Value)
        d_k: key的维度
    Args:
        dropout: 注意力权重的dropout概率
    shape:
        - Q: (batch_size, num_attention_heads, seq_length_q, d_k)
        - K: (batch_size, num_attention_heads, seq_length_k, d_k)
        - V: (batch_size, num_attention_heads, seq_length_v, d_v)
        - mask: (batch_size, seq_length_q, seq_length_k) 或者 (batch_size, 1, seq_length_k)
        - Output: (batch_size, num_attention_heads, seq_length_q, d_v)
        - Attention Weights: (batch_size, num_attention_heads, seq_length_q, seq_length_k)
    注意:
        通常 seq_length_k == seq_length_v 因为每个键对应一个值
        对于自注意力机制，seq_length_q == seq_length_k == seq_length_v
        d_k == d_v == embedding_dim / num_attention_heads, 通常相同, 但是也可以不同
    """

    def __init__(self, dropout_rate: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算注意力输出
        Args:
            query: 查询矩阵 (batch_size, num_attention_heads, seq_length_q, d_k)
            key: 键矩阵 (batch_size, num_attention_heads, seq_length_k, d_k)
            value: 值矩阵 (batch_size, num_attention_heads, seq_length_v, d_v)
            mask: 可选的mask矩阵 (batch_size, seq_length_q, seq_length_k) 或者 (batch_size, 1, seq_length_k)
            return_attention_weights: 是否返回注意力权重
        Returns:
            if return_attention_weights:
                output: 注意力输出 (batch_size, num_attention_heads, seq_length_q, d_v)
                attention_weights: 注意力权重 (batch_size, num_attention_heads, seq_length_q, seq_length_k)
            else:
                output: 注意力输出 (batch_size, num_attention_heads, seq_length_q, d_v)
        """
        d_k = query.size(-1)  # 获取key的维度
        # step1: 计算点积注意力分数
        # query: (batch_size, num_attention_heads, seq_length_q, d_k)
        # key 矩阵需要转置, 所以需要把最后两个维度交换: key.transpose(-2, -1): (batch_size, num_attention_heads, d_k, seq_length_k)
        # key.transpose(-2, -1): (batch_size, num_attention_heads, d_k, seq_length_k)
        # scores: (batch_size, num_attention_heads, seq_length_q, seq_length_k)
        # 除以 sqrt(d_k) 进行缩放, 防止点积过大导致softmax梯度消失
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )

        # step2: 如果提供了mask，则将mask应用于分数
        if mask is not None:
            # 如果 mask 是 3-dim (batch_size, seq_length_q, seq_length_k)，则需要扩展到 4-dim (batch_size, 1, seq_length_q, seq_length_k) 以匹配 scores 的维度
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_length_q, seq_length_k)

            # 将 mask 为 True 的位置的分数设为负无穷，这样 softmax 后对应位置的权重为0
            scores = scores.masked_fill(
                mask, float("-inf")
            )  # 将mask位置的分数设为负无穷

        # step3: 计算注意力权重, 对分数进行softmax归一化
        # 在最后一个维度上进行softmax, 因为我们要对每个查询对应的所有键进行归一化
        # attention_weights: (batch_size, num_attention_heads, seq_length_q, seq_length_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)  # 应用dropout

        # step4: 计算加权值, attention_weights * value
        # attention_weights: (batch_size, num_attention_heads, seq_length_q, seq_length_k)
        # value: (batch_size, num_attention_heads, seq_length_v, d_v), 其中 seq_length_v 通常等于 seq_length_k
        # output: (batch_size, num_attention_heads, seq_length_q, d_v)
        output = torch.matmul(attention_weights, value)

        if return_attention_weights:
            return output, attention_weights
        return output


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Scaled Dot-Product Attention")
    print("=" * 60)

    # 设置参数
    batch_size = 2
    num_heads = 8
    seq_len = 10
    d_k = 64  # key/query 维度
    d_v = 64  # value 维度

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}\n")

    # 创建模型
    attention = ScaledDotProductAttention(dropout_rate=0.1).to(device)
    print(f"模型参数量: {sum(p.numel() for p in attention.parameters())}")
    print("(ScaledDotProductAttention 没有可训练参数，只有 dropout)\n")

    # 1. 测试基本前向传播
    print("1. 基本前向传播")
    print("-" * 60)

    # 创建输入
    query = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, d_v, device=device)

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")

    # 前向传播
    output = attention(query, key, value)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, num_heads, seq_len, d_v), "输出形状错误"
    print("✓ 基本前向传播测试通过\n")

    # 2. 测试带 mask 的注意力
    print("2. 带 mask 的注意力")
    print("-" * 60)

    # 创建因果掩码（自回归）
    from src.model.mask import create_causal_mask

    causal_mask = create_causal_mask(seq_len, device=device)
    print(f"Causal mask shape: {causal_mask.shape}")  # (seq_len, seq_len)

    # 扩展到 batch 维度
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    print(f"Expanded mask shape: {causal_mask.shape}")  # (batch_size, seq_len, seq_len)

    # 前向传播
    output_with_mask = attention(query, key, value, mask=causal_mask)
    print(f"Output with mask shape: {output_with_mask.shape}")
    assert output_with_mask.shape == (batch_size, num_heads, seq_len, d_v), (
        "带mask的输出形状错误"
    )
    print("✓ 带 mask 的注意力测试通过\n")

    # 3. 测试返回注意力权重
    print("3. 返回注意力权重")
    print("-" * 60)

    attention.eval()  # 关闭 dropout 以便查看权重
    with torch.no_grad():
        output, attn_weights = attention(
            query, key, value, mask=causal_mask, return_attention_weights=True
        )

    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), (
        "注意力权重形状错误"
    )

    # 验证注意力权重的性质
    # 1. 每一行的权重和应该为 1（softmax 归一化）
    row_sums = attn_weights.sum(dim=-1)
    print(f"Attention weights 每行求和（应该全为1）: {row_sums[0, 0, :5]}")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), (
        "注意力权重每行求和不为1"
    )

    # 2. 因果mask的位置权重应该为 0
    # 检查第一个样本的第一个头
    first_head_weights = attn_weights[0, 0]  # (seq_len, seq_len)
    print("\n第一个头的注意力权重矩阵（前5x5）:")
    print(first_head_weights[:5, :5])
    print("注意：上三角（未来位置）的权重应该为 0\n")

    # 验证上三角为 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert first_head_weights[i, j] < 1e-6, (
                f"位置 ({i}, {j}) 的权重应该为0，但实际为 {first_head_weights[i, j]}"
            )

    print("✓ 注意力权重测试通过\n")

    # 4. 测试不同的 seq_len (cross-attention)
    print("4. 测试 Cross-Attention（不同序列长度）")
    print("-" * 60)

    seq_len_q = 8  # query 序列长度
    seq_len_kv = 12  # key/value 序列长度

    query_cross = torch.randn(batch_size, num_heads, seq_len_q, d_k, device=device)
    key_cross = torch.randn(batch_size, num_heads, seq_len_kv, d_k, device=device)
    value_cross = torch.randn(batch_size, num_heads, seq_len_kv, d_v, device=device)

    print(f"Query shape: {query_cross.shape}")
    print(f"Key shape: {key_cross.shape}")
    print(f"Value shape: {value_cross.shape}")

    output_cross = attention(query_cross, key_cross, value_cross)
    print(f"Output shape: {output_cross.shape}")
    assert output_cross.shape == (batch_size, num_heads, seq_len_q, d_v), (
        "Cross-attention 输出形状错误"
    )
    print("✓ Cross-Attention 测试通过\n")

    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
