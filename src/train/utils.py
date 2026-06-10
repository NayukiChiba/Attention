"""
src/train/train_utils.py

训练工具函数
"""

import torch
import torch.nn as nn


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    梯度裁剪

    Args:
        model (nn.Module): 模型
        max_norm (float): 最大梯度范数

    Returns:
        float: 裁剪前的总梯度范数
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数量

    Args:
        model (nn.Module): 模型
        trainable_only (bool): 是否只统计可训练参数，默认 True

    Returns:
        int: 参数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print("测试训练工具函数")
    print("=" * 60)

    from config.defaults import GPTConfig
    from src.model.gpt import GPT

    # 创建测试模型
    config = GPTConfig(
        vocab_size=1000,
        context_length=128,
        embedding_dim=256,
        num_attention_heads=4,
        num_layers=4,
    )

    model = GPT(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 1. 测试参数统计
    print("\n1. 测试参数统计")
    print("-" * 60)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")

    # 2. 测试梯度裁剪
    print("\n2. 测试梯度裁剪")
    print("-" * 60)

    # 创建输入和目标
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )
    target_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    # 前向传播
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        target_ids.view(-1),
    )

    # 反向传播
    loss.backward()

    # 梯度裁剪（返回裁剪前的梯度范数）
    max_norm = 1.0
    grad_norm_before = clip_gradients(model, max_norm)

    print(f"裁剪前梯度范数: {grad_norm_before:.4f}")
    print(f"最大范数限制: {max_norm:.4f}")

    # 计算裁剪后的实际梯度范数
    grad_norm_after = torch.sqrt(
        sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
    ).item()

    print(f"裁剪后梯度范数: {grad_norm_after:.4f}")

    # 验证裁剪是否有效
    if grad_norm_before > max_norm:
        assert grad_norm_after <= max_norm + 1e-6, "梯度裁剪失败"
        print("梯度已裁剪")
    else:
        print("梯度未超过限制，无需裁剪")

    # 3. 测试冻结部分参数
    print("\n3. 测试冻结部分参数")
    print("-" * 60)

    # 冻结 embedding 层
    for param in model.embedding.parameters():
        param.requires_grad = False

    total_params_frozen = count_parameters(model, trainable_only=False)
    trainable_params_frozen = count_parameters(model, trainable_only=True)

    print(f"总参数量: {total_params_frozen:,}")
    print(f"可训练参数量: {trainable_params_frozen:,}")
    print(f"冻结参数量: {total_params_frozen - trainable_params_frozen:,}")

    assert trainable_params_frozen < trainable_params, "参数冻结失败"
    print("参数冻结测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
