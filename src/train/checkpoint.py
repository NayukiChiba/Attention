"""
src/train/checkpoint.py

模型检查点保存和加载函数
"""

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    loss: float,
    checkpoint_path: Path,
    **kwargs,
) -> None:
    """
    保存模型检查点

    Args:
        model (nn.Module): 模型
        optimizer (optim.Optimizer): 优化器
        scheduler: 学习率调度器(可以为 None)
        epoch (int): 当前训练轮数
        step (int): 当前训练步数
        loss (float): 当前损失值
        checkpoint_path (Path): 检查点保存路径
        **kwargs: 其他需要保存的信息
    """
    # 确保目录存在
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # 构建检查点字典
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # 保存调度器状态(如果存在)
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # 保存额外信息
    checkpoint.update(kwargs)

    # 保存到文件
    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存到: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler: Any = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    加载模型检查点

    Args:
        checkpoint_path (Path): 检查点文件路径
        model (nn.Module): 模型
        optimizer (optim.Optimizer): 优化器(可选)
        scheduler: 学习率调度器(可选)
        device (str): 设备

    Returns:
        Dict[str, Any]: 检查点字典,包含 epoch, step, loss 等信息
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"])

    # 加载优化器状态(如果提供)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 加载调度器状态(如果提供)
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"检查点已加载: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")

    return checkpoint
