"""
src/model/transformerBlock.py

Transformer Block 的实现


"""

import torch
import torch.nn as nn

from src.model.feedForward import FeedForward
from src.model.multiHeadAttention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Transformer Block 的实现

    结构 (Pre-Norm)
        1. x = x + MultiHeadAttention(LayerNorm(x))
        2. x = x + FeedForward(LayerNorm(x))

    结构 (Post-Norm)
        1. x = LayerNorm(x + MultiHeadAttention(x))
        2. x = LayerNorm(x + FeedForward(x))
    Args:

        embedding_dim (int): 模型的维度
        num_attention_heads (int): 注意力头的数量
        ffn_hidden_dim (int): 前馈网络的维度
        dropout (float): Dropout 概率
        activation (str): 前馈网络的激活函数 ("relu" 或 "gelu")
        norm_type (str): LayerNorm 的位置 ("pre" 或 "post")
        layer_norm_eps (float): LayerNorm 的数值稳定项


    """

    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int,
        ffn_hidden_dim: int,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        norm_type: str = "pre",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        assert norm_type in ["pre", "post"], "norm_type must be 'pre' or 'post'"
        self.norm_type = norm_type

        # 多头注意力层
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout_rate=dropout_rate,
        )

        # 前馈网络层
        self.ffn = FeedForward(
            embedding_dim=embedding_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout_rate=dropout_rate,
            activation=activation,
        )

        # LayerNorm 层
        self.norm1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)

        # dropout层（可选）
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播计算Transformer Block的输出
        Args:
            x: 输入张量 (batch_size, seq_length, embedding_dim)
            mask: 可选的mask张量 (batch_size, seq_length_q, seq_length_k)
        Returns:
            输出张量 (batch_size, seq_length, embedding_dim)
        """
        if self.norm_type == "pre":
            # Pre-Norm: LayerNorm -> SubLayer -> Residual
            # 多头注意力 + 残差连接
            normed_x = self.norm1(x)
            attention_output = self.attention(
                query=normed_x, key=normed_x, value=normed_x, mask=mask
            )
            x = x + attention_output

            # 前馈网络 + 残差连接
            normed_x = self.norm2(x)
            ffn_output = self.ffn(normed_x)
            x = x + self.dropout(ffn_output)
        else:
            # Post-Norm: SubLayer -> Residual -> LayerNorm
            # Post-Norm 结构
            attention_output = self.attention(query=x, key=x, value=x, mask=mask)
            x = self.norm1(x + self.dropout(attention_output))
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_output))

        return x


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Transformer Block")
    print("=" * 60)

    # 设置参数
    batch_size = 2
    seq_len = 10
    embedding_dim = 512
    num_attention_heads = 8
    ffn_hidden_dim = 2048
    dropout_rate = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}\n")

    # 1. 测试 Pre-Norm
    print("1. 测试 Pre-Norm")
    print("-" * 60)

    block_pre = TransformerBlock(
        embedding_dim=embedding_dim,
        num_attention_heads=num_attention_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout_rate=dropout_rate,
        activation="gelu",
        norm_type="pre",
    ).to(device)

    print("模型参数:")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  num_attention_heads: {num_attention_heads}")
    print(f"  ffn_hidden_dim: {ffn_hidden_dim}")
    print("  norm_type: pre")
    print(f"  总参数量: {sum(p.numel() for p in block_pre.parameters()):,}")

    # 创建输入
    x = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    print(f"\nInput shape: {x.shape}")

    # 前向传播
    output_pre = block_pre(x)
    print(f"Output shape: {output_pre.shape}")
    assert output_pre.shape == (batch_size, seq_len, embedding_dim), "输出形状错误"
    print("✓ Pre-Norm 测试通过\n")

    # 2. 测试 Post-Norm
    print("2. 测试 Post-Norm")
    print("-" * 60)

    block_post = TransformerBlock(
        embedding_dim=embedding_dim,
        num_attention_heads=num_attention_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout_rate=dropout_rate,
        activation="gelu",
        norm_type="post",
    ).to(device)

    print("模型参数:")
    print("  norm_type: post")

    output_post = block_post(x)
    print(f"Output shape: {output_post.shape}")
    assert output_post.shape == (batch_size, seq_len, embedding_dim), "输出形状错误"
    print("✓ Post-Norm 测试通过\n")

    # 3. 测试带 mask
    print("3. 测试带 mask")
    print("-" * 60)

    from src.model.mask import create_causal_mask

    causal_mask = create_causal_mask(seq_len, device=device)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    print(f"Causal mask shape: {causal_mask.shape}")

    output_with_mask = block_pre(x, mask=causal_mask)
    print(f"Output with mask shape: {output_with_mask.shape}")
    assert output_with_mask.shape == (batch_size, seq_len, embedding_dim), (
        "带mask的输出形状错误"
    )
    print("✓ 带 mask 测试通过\n")

    # 4. 测试梯度反向传播
    print("4. 测试梯度反向传播")
    print("-" * 60)

    block_pre.train()
    x_grad = torch.randn(
        batch_size, seq_len, embedding_dim, device=device, requires_grad=True
    )

    output_grad = block_pre(x_grad, mask=causal_mask)
    loss = output_grad.sum()
    loss.backward()

    print(f"Input 梯度存在: {x_grad.grad is not None}")
    print(f"attention 参数梯度存在: {block_pre.attention.W_Q.weight.grad is not None}")
    print(f"ffn 参数梯度存在: {block_pre.ffn.linear1.weight.grad is not None}")
    print(f"norm1 参数梯度存在: {block_pre.norm1.weight.grad is not None}")
    assert x_grad.grad is not None, "输入梯度不存在"
    print("✓ 梯度反向传播测试通过\n")

    # 5. 测试残差连接的效果
    print("5. 测试残差连接的效果")
    print("-" * 60)

    block_pre.eval()
    with torch.no_grad():
        # 测试输入接近零的情况
        x_zero = torch.zeros(batch_size, seq_len, embedding_dim, device=device)
        output_zero = block_pre(x_zero)

        # 残差连接应该让输出不全为零
        print(f"零输入的输出均值: {output_zero.mean().item():.6f}")
        print(f"零输入的输出标准差: {output_zero.std().item():.6f}")

        # 测试正常输入
        x_normal = torch.randn(batch_size, seq_len, embedding_dim, device=device)
        output_normal = block_pre(x_normal)

        # 计算输入和输出的差异（残差应该相对较小）
        residual = output_normal - x_normal
        print(f"\n正常输入的残差均值: {residual.mean().item():.6f}")
        print(f"正常输入的残差标准差: {residual.std().item():.6f}")

    print("✓ 残差连接测试通过\n")

    # 6. 测试不同的 activation
    print("6. 测试不同的 activation")
    print("-" * 60)

    for act in ["gelu", "relu"]:
        block_act = TransformerBlock(
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout_rate=dropout_rate,
            activation=act,
            norm_type="pre",
        ).to(device)

        output_act = block_act(x)
        print(f"activation={act}: 输出形状={output_act.shape}")
        assert output_act.shape == (batch_size, seq_len, embedding_dim), "输出形状错误"

    print("✓ 不同 activation 测试通过\n")

    # 7. 测试堆叠多个 Block
    print("7. 测试堆叠多个 Block")
    print("-" * 60)

    num_layers = 4
    blocks = nn.ModuleList(
        [
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_attention_heads=num_attention_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout_rate=dropout_rate,
                activation="gelu",
                norm_type="pre",
            ).to(device)
            for _ in range(num_layers)
        ]
    )

    x_stacked = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    print(f"堆叠 {num_layers} 个 Block")
    print(f"Input shape: {x_stacked.shape}")

    for i, block in enumerate(blocks):
        x_stacked = block(x_stacked, mask=causal_mask)
        print(f"  Block {i + 1} output shape: {x_stacked.shape}")

    assert x_stacked.shape == (batch_size, seq_len, embedding_dim), "堆叠输出形状错误"
    print("✓ 堆叠多个 Block 测试通过\n")

    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
