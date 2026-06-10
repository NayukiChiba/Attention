"""
src/evaluate/evaluator.py

模型评估器
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.defaults import TrainingConfig


class Evaluator:
    """
    模型评估器

    使用方式:
        evaluator = Evaluator(model, test_loader, config)
        metrics = evaluator.evaluate()
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: TrainingConfig,
    ):
        """
        初始化评估器

        Args:
            model: 模型
            test_loader: 测试集数据加载器
            config: 训练配置
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config

    def evaluate(self) -> Dict[str, float]:
        """
        在测试集上评估模型

        Returns:
            Dict[str, float]: 评估指标 {"test_loss": xxx, "test_ppl": xxx}
        """

        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.test_loader)

        pbar = tqdm(self.test_loader, desc="[TEST] Evaluating")

        with torch.no_grad():
            for batch in pbar:
                # 获取输入和目标（dataset 返回 (input_ids, target_ids) 元组）
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.config.device)
                target_ids = target_ids.to(self.config.device)

                # 前向传播
                logits = self.model(input_ids)

                # 计算损失
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                )

                total_loss += loss.item()

                # 计算困惑度
                ppl = torch.exp(loss).item()

                # 更新进度条
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "ppl": f"{ppl:.4f}",
                    }
                )

        avg_loss = total_loss / num_batches
        avg_ppl = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {
            "test_loss": avg_loss,
            "test_ppl": avg_ppl,
        }

        return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("测试评估器")
    print("=" * 60)

    from config.defaults import GPTConfig, TrainingConfig
    from src.model.gpt import GPT

    # 创建测试模型
    gpt_config = GPTConfig(
        vocab_size=1000,
        context_length=128,
        embedding_dim=256,
        num_attention_heads=4,
        num_layers=4,
    )

    model = GPT(gpt_config)

    training_config = TrainingConfig()

    print("\n评估器创建成功")
    print("提示：需要 test_loader 才能运行完整测试")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
