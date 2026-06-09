"""
src/model/mask.py

生成注意力掩码(Attention Mask)的函数实现
"""

import torch


def create_causal_mask(
    seq_length: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    创建一个因果掩码(Causal Mask),用于GPT模型的自回归注意力机制

    Args:
        seq_length (int): 序列长度
        device (str): 设备类型,默认为 "cuda"(如果可用)或 "cpu"

    Returns:
        torch.Tensor: 形状为 (1, seq_length, seq_length) 的因果掩码张量
        允许注意力表示为0, 禁止注意力表示为负无穷大
        True 表示允许注意力, 需要被mask, 不能被 attention 到
        False 表示不需要被mask, 可以被 attention 到
    Example:
        >>> mask = create_causal_mask(5)
        >>> print(mask)
        tensor([[[False,  True,  True,  True,  True],
                 [False, False,  True,  True,  True],
                 [False, False, False,  True,  True],
                 [False, False, False, False,  True],
                 [False, False, False, False, False]]])

        位置0 只能看到位置 0
        位置1 可以看到位置 0, 1
        位置2 可以看到位置 0, 1, 2
        位置3 可以看到位置 0, 1, 2, 3
        位置4 可以看到位置 0, 1, 2, 3, 4
    """
    # 创建一个上三角矩阵,包含True(允许注意力)和False(禁止注意力)
    mask = torch.triu(
        torch.ones((seq_length, seq_length), dtype=torch.bool, device=device),
        diagonal=1,
    ).to(device)
    # 添加一个批次维度,形状变为 (seq_length, seq_length)
    return mask


def create_padding_mask(
    input_ids: torch.Tensor,
    pad_token_id: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    创建一个填充掩码(Padding Mask),用于标记输入序列中的填充位置

    由于我们在数据集中使用 <PAD> token 来填充短文本,模型不应该关注这些位置
    填充掩码通常与注意力掩码结合使用,确保模型不会将注意力分配给填充位置

    Args:
        input_ids (torch.Tensor): 输入的 token id 张量, 形状为 (batch_size, seq_length)
        pad_token_id (int): 用于填充的 token id,通常是 tokenizer 的 pad_token_id
        device (str): 设备类型,默认为 "cuda"(如果可用)或 "cpu"

    Returns:
        torch.Tensor: 形状为 (batch_size, seq_length) 的填充掩码张量
        True 表示该位置是填充, 需要被mask, 不能被 attention 到
        False 表示该位置不是填充, 不需要被mask, 可以被 attention 到

    Example:
        >>> input_ids = torch.tensor([[1, 2, 3, 0, 0],  # 0是padding
        ...                           [4, 5, 0, 0, 0]])
        >>> mask = create_padding_mask(input_ids, pad_token_id=0)
        >>> mask
        tensor([[False, False, False,  True,  True],
                [False, False,  True,  True,  True]])
    """
    # 这里我们暂时返回一个全False的掩码,因为我们在数据集中已经处理了文本长度
    # 如果你在后续实现中使用了动态长度的输入,可以根据实际情况生成填充掩码
    # shape=(batch_size, seq_length)
    return (input_ids == pad_token_id).to(device)


def combine_masks(
    causal_mask: torch.Tensor, padding_mask: torch.Tensor
) -> torch.Tensor:
    """
    组合因果掩码和填充掩码,生成最终的注意力掩码
    Args:
        causal_mask (torch.Tensor): 因果掩码,形状为 (seq_length, seq_length)
        padding_mask (torch.Tensor): 填充掩码,形状为 (batch_size, seq_length)
    Returns:
        torch.Tensor: 形状为 (batch_size, seq_length, seq_length) 的最终注意力掩码
        True 表示该位置需要被mask, 不能被 attention 到
        False 表示该位置不需要被mask, 可以被 attention 到
    """
    if padding_mask is None:
        # 没有 padding_mask, 直接使用 causal_mask
        # shape=(1, seq_length, seq_length)
        return causal_mask.unsqueeze(0)
    # padding_mask 需要扩展维度以匹配 causal_mask 的形状: (batch_size, seq_length) -> (batch_size, 1, seq_length)
    padding_mask = padding_mask.unsqueeze(1)

    # 广播并且组合两个掩码: 只要其中一个是 True 就表示需要被mask
    # (1, seq_length, seq_length) 和 (batch_size, 1, seq_length) 广播后形状为 (batch_size, seq_length, seq_length)
    causal_mask = causal_mask.unsqueeze(0)  # shape=(1, seq_length, seq_length)
    combined_mask = causal_mask | padding_mask

    return combined_mask


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Mask 生成")
    print("=" * 60)

    # 测试因果掩码
    print("\n1. 因果掩码(Causal Mask)")
    print("-" * 60)
    causal_mask = create_causal_mask(5)
    print(f"形状: {causal_mask.shape}")  # (5, 5)
    print(f"内容:\n{causal_mask}")

    # 验证:对角线及以下应该全是False
    assert not causal_mask[0, 0], "对角线应该是False"
    assert causal_mask[0, 1], "对角线以上应该是True"
    print("✓ 因果掩码测试通过")

    # 测试填充掩码
    print("\n2. 填充掩码(Padding Mask)")
    print("-" * 60)
    # shape: (2, 6)
    input_ids = torch.tensor(
        [
            [2, 10, 20, 30, 0, 0],  # 最后2个是padding
            [2, 15, 25, 0, 0, 0],  # 最后3个是padding
        ]
    )
    padding_mask = create_padding_mask(input_ids, pad_token_id=0)
    print(f"输入 token ids 形状: {input_ids.shape}")  # (2, 6)
    print(f"输入 token ids:\n{input_ids}")
    print(f"填充掩码形状: {padding_mask.shape}")  # (2, 6)
    print(f"填充掩码:\n{padding_mask}")
    assert padding_mask[0, 4], "padding位置应该是True"
    assert not padding_mask[0, 0], "非padding位置应该是False"
    print("✓ 填充掩码测试通过")

    # 测试组合掩码
    print("\n3. 组合掩码(Combined Mask)")
    print("-" * 60)
    causal_mask = create_causal_mask(6)  # (6, 6)
    combined_mask = combine_masks(causal_mask, padding_mask)  # (2, 6, 6)
    print(f"因果掩码形状: {causal_mask.shape}")  # (6, 6)
    print(f"填充掩码形状: {padding_mask.shape}")  # (2, 6)
    print(f"组合掩码形状: {combined_mask.shape}")  # (2, 6, 6)
    print(f"第一个样本的组合掩码:\n{combined_mask[0]}")
    print(f"第二个样本的组合掩码:\n{combined_mask[1]}")

    # 验证:padding位置的整列都应该是True
    assert combined_mask[0, 0, 4], "padding位置应该被mask"
    assert not combined_mask[0, 2, 1], "非padding且在当前位置之前的位置应该不被mask"

    print("✓ 组合掩码测试通过")

    # 测试在attention中的使用
    print("\n4. 在 Attention 中的使用示例")
    print("-" * 60)
    batch_size, seq_len, d_model = 2, 4, 8

    # 获取设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟attention scores
    # shape: (batch_size, seq_len, seq_len)
    scores = torch.randn(batch_size, seq_len, seq_len, device=device)
    print(f"原始 attention scores 形状: {scores.shape}")
    print(f"设备: {scores.device}")

    # 创建mask
    mask = create_causal_mask(seq_len, device=device)  # (seq_len, seq_len)
    mask = mask.unsqueeze(0).expand(
        batch_size, -1, -1
    )  # (batch_size, seq_len, seq_len)
    print(f"扩展后的mask形状: {mask.shape}")

    # 应用mask:将masked位置设为负无穷
    scores_masked = scores.masked_fill(mask, float("-inf"))

    print(f"\n应用mask后的scores(第一个样本):\n{scores_masked[0]}")
    print("注意:对角线以上的位置变成了 -inf")

    # 应用softmax
    import torch.nn.functional as F

    attention_weights = F.softmax(scores_masked, dim=-1)
    print(f"\nSoftmax后的attention权重形状: {attention_weights.shape}")  # (2, 4, 4)
    print(f"Softmax后的attention权重(第一个样本):\n{attention_weights[0]}")
    print("注意:-inf位置的权重变成了0")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
