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
    # 添加一个批次维度,形状变为 (1, seq_length, seq_length)
    return mask


def create_padding_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    创建一个填充掩码(Padding Mask),用于标记输入序列中的填充位置

    由于我们在数据集中使用 <PAD> token 来填充短文本,模型不应该关注这些位置
    填充掩码通常与注意力掩码结合使用,确保模型不会将注意力分配给填充位置

    Args:
        input_ids (torch.Tensor): 输入的 token id 张量, 形状为 (batch_size, seq_length)
        pad_token_id (int): 用于填充的 token id,通常是 tokenizer 的 pad_token_id

    Returns:
        torch.Tensor: 形状为 (batch_size, 1, seq_length) 的填充掩码张量
        True 表示该位置是填充, 需要被mask, 不能被 attention 到
        False 表示该位置不是填充, 不需要被mask, 可以被 attention 到
    """
    # 这里我们暂时返回一个全False的掩码,因为我们在数据集中已经处理了文本长度
    # 如果你在后续实现中使用了动态长度的输入,可以根据实际情况生成填充掩码
    # shape=(batch_size, 1, seq_length)
    return input_ids == pad_token_id
