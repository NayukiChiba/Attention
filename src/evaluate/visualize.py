"""
src/evaluate/visualize.py

训练可视化
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制训练和验证的 loss 和 perplexity 曲线

    Args:
        history: 训练历史字典，包含 train_loss, train_ppl, val_loss, val_ppl
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # 绘制 Loss 曲线
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 绘制 Perplexity 曲线
    axes[1].plot(epochs, history["train_ppl"], label="Train PPL", color="blue")
    axes[1].plot(epochs, history["val_ppl"], label="Val PPL", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Training and Validation Perplexity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    tokens: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制注意力权重热力图

    Args:
        attention_weights: 注意力权重矩阵，形状 (seq_len, seq_len)
        tokens: token 列表（可选）
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制热力图
    im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")

    # 设置坐标轴
    if tokens:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticklabels(tokens)

    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title("Attention Weights Heatmap")

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight", rotation=270, labelpad=15)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"注意力热力图已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("测试可视化功能")
    print("=" * 60)

    from config import paths

    # 1. 测试训练曲线
    print("\n1. 测试训练曲线")
    print("-" * 60)

    # 模拟训练历史
    history = {
        "train_loss": [2.5, 2.0, 1.5, 1.2, 1.0, 0.9, 0.85, 0.8],
        "train_ppl": [12.2, 7.4, 4.5, 3.3, 2.7, 2.5, 2.3, 2.2],
        "val_loss": [2.6, 2.1, 1.6, 1.3, 1.1, 1.0, 0.95, 0.92],
        "val_ppl": [13.5, 8.2, 5.0, 3.7, 3.0, 2.7, 2.6, 2.5],
    }

    save_path = paths.FIGURES_DIR / "training_curves.png"
    plot_training_curves(history, save_path)

    # 2. 测试注意力热力图
    print("\n2. 测试注意力热力图")
    print("-" * 60)

    # 模拟注意力权重
    seq_len = 8
    attention_weights = np.random.rand(seq_len, seq_len)
    # 归一化
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

    tokens = ["今", "天", "天", "气", "很", "好", "<EOS>", "<PAD>"]

    save_path = paths.FIGURES_DIR / "attention_heatmap.png"
    plot_attention_heatmap(attention_weights, tokens, save_path)

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
