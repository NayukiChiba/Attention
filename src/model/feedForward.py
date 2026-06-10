"""
src/model/feedForward.py

前馈神经网络的实现(Feed-Forward Network)


"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    前馈神经网络 (Position-wise Feed-Forward Network) 的实现

    Transformer 中的 FFN 结构
    计算公式:
        FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2

    Args:
        embedding_dim: 输入和输出的维度 (通常与Transformer的embedding_dim相同)
        ffn_hidden_dim: FFN中间层的维度 (通常大于embedding_dim, 如4倍)
        dropout_rate: FFN中间层的dropout概率
        activation: FFN中间层的激活函数 (默认ReLU)
    """

    def __init__(
        self,
        embedding_dim: int,
        ffn_hidden_dim: int,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        # 线性层1: 从embedding_dim映射到ffn_hidden_dim
        self.linear1 = nn.Linear(embedding_dim, ffn_hidden_dim)

        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.dropout = nn.Dropout(dropout_rate)

        # 线性层2: 从ffn_hidden_dim映射回embedding_dim
        self.linear2 = nn.Linear(ffn_hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算FFN输出
        Args:
            x: 输入张量 (batch_size, seq_length, embedding_dim)
        Returns:
            output: FFN输出 (batch_size, seq_length, embedding_dim)
        """
        # 线性层1 + 激活函数 + dropout
        hidden = self.linear1(x)  # (batch_size, seq_length, ffn_hidden_dim)
        hidden = self.activation(hidden)  # (batch_size, seq_length, ffn_hidden_dim)
        hidden = self.dropout(hidden)  # (batch_size, seq_length, ffn_hidden_dim)

        # 线性层2
        output = self.linear2(hidden)  # (batch_size, seq_length, embedding_dim)

        return output


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Feed-Forward Network")
    print("=" * 60)

    # 设置参数
    batch_size = 2
    seq_len = 10
    embedding_dim = 512
    ffn_hidden_dim = 2048  # 通常是 embedding_dim 的 4 倍
    dropout_rate = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}\n")

    # 1. 测试 GELU 激活函数
    print("1. 测试 GELU 激活函数")
    print("-" * 60)

    ffn_gelu = FeedForward(
        embedding_dim=embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout_rate=dropout_rate,
        activation="gelu",
    ).to(device)

    print("模型参数:")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  ffn_hidden_dim: {ffn_hidden_dim}")
    print("  activation: gelu")
    print(f"  总参数量: {sum(p.numel() for p in ffn_gelu.parameters()):,}")

    # 创建输入
    x = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    print(f"\nInput shape: {x.shape}")

    # 前向传播
    output = ffn_gelu(x)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, embedding_dim), "输出形状错误"
    print("✓ GELU 激活函数测试通过\n")

    # 2. 测试 ReLU 激活函数
    print("2. 测试 ReLU 激活函数")
    print("-" * 60)

    ffn_relu = FeedForward(
        embedding_dim=embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout_rate=dropout_rate,
        activation="relu",
    ).to(device)

    print("模型参数:")
    print("  activation: relu")

    output_relu = ffn_relu(x)
    print(f"Output shape: {output_relu.shape}")
    assert output_relu.shape == (batch_size, seq_len, embedding_dim), "输出形状错误"
    print("✓ ReLU 激活函数测试通过\n")

    # 3. 测试梯度反向传播
    print("3. 测试梯度反向传播")
    print("-" * 60)

    ffn_gelu.train()
    x_grad = torch.randn(
        batch_size, seq_len, embedding_dim, device=device, requires_grad=True
    )

    output_grad = ffn_gelu(x_grad)
    loss = output_grad.sum()
    loss.backward()

    print(f"Input 梯度存在: {x_grad.grad is not None}")
    print(f"linear1 权重梯度存在: {ffn_gelu.linear1.weight.grad is not None}")
    print(f"linear2 权重梯度存在: {ffn_gelu.linear2.weight.grad is not None}")
    assert x_grad.grad is not None, "输入梯度不存在"
    assert ffn_gelu.linear1.weight.grad is not None, "linear1 梯度不存在"
    assert ffn_gelu.linear2.weight.grad is not None, "linear2 梯度不存在"
    print("✓ 梯度反向传播测试通过\n")

    # 4. 测试不同的 ffn_hidden_dim 比例
    print("4. 测试不同的 ffn_hidden_dim 比例")
    print("-" * 60)

    for ratio in [2, 4, 8]:
        ffn_ratio = FeedForward(
            embedding_dim=embedding_dim,
            ffn_hidden_dim=embedding_dim * ratio,
            dropout_rate=dropout_rate,
            activation="gelu",
        ).to(device)

        output_ratio = ffn_ratio(x)
        params = sum(p.numel() for p in ffn_ratio.parameters())

        print(
            f"ratio={ratio}x: ffn_hidden_dim={embedding_dim * ratio}, 参数量={params:,}, 输出形状={output_ratio.shape}"
        )
        assert output_ratio.shape == (batch_size, seq_len, embedding_dim), (
            "输出形状错误"
        )

    print("✓ 不同比例测试通过\n")

    # 5. 测试 eval 模式（dropout 关闭）
    print("5. 测试 eval 模式（dropout 关闭）")
    print("-" * 60)

    ffn_gelu.eval()
    with torch.no_grad():
        output_eval1 = ffn_gelu(x)
        output_eval2 = ffn_gelu(x)

    # eval 模式下，相同输入应该产生相同输出
    print(f"两次前向传播结果是否相同: {torch.allclose(output_eval1, output_eval2)}")
    assert torch.allclose(output_eval1, output_eval2), "eval 模式下输出不一致"
    print("✓ eval 模式测试通过\n")

    # 6. 测试 train 模式（dropout 开启）
    print("6. 测试 train 模式（dropout 开启）")
    print("-" * 60)

    ffn_gelu.train()
    with torch.no_grad():
        output_train1 = ffn_gelu(x)
        output_train2 = ffn_gelu(x)

    # train 模式下，由于 dropout 的随机性，输出应该不同
    print(
        f"两次前向传播结果是否不同: {not torch.allclose(output_train1, output_train2)}"
    )
    assert not torch.allclose(output_train1, output_train2), "train 模式下输出应该不同"
    print("✓ train 模式测试通过\n")

    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
