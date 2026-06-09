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
