"""
src/train/optimizer.py

优化器创建函数

"""

import torch.nn as nn
import torch.optim as optim

from config.defaults import TrainingConfig


def create_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """
    创建优化器

    Args:
        model (nn.Module): 模型
        config (TrainingConfig): 训练配置

    Returns:
        optim.Optimizer: 优化器
    """
    if config.optimizer_type == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.betas[0],  # SGD使用momentum参数，取betas的第一个值
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    return optimizer
