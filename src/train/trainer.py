"""
src/train/trainer.py

训练器
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import paths
from config.defaults import TrainingConfig
from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.train.early_stopping import EarlyStopping
from src.train.logger import Logger
from src.train.optimizer import create_optimizer
from src.train.scheduler import create_scheduler
from src.train.utils import clip_gradients, count_parameters


class Trainer:
    """
    训练器

    使用方式:
        trainer = Trainer(model, train_loader, val_loader, config, checkpoint_dir, log_dir, tensorboard_dir)
        history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        checkpoint_dir: Path,
        log_dir: Path,
        tensorboard_dir: Path,
    ):
        """
        初始化训练器

        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
            checkpoint_dir: 检查点保存目录
            log_dir: 日志目录
            tensorboard_dir: TensorBoard 日志目录
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.tensorboard_dir = tensorboard_dir

        # 初始化日志记录器
        self.logger = Logger(
            name="training",
            log_dir=log_dir,
            tensorboard_dir=tensorboard_dir,
        )
        self.logger.start()

        # 记录配置
        self.logger.log_config(config)

        # 统计参数量
        total_params = count_parameters(model, trainable_only=False)
        trainable_params = count_parameters(model, trainable_only=True)
        self.logger.info(f"总参数量: {total_params:,}")
        self.logger.info(f"可训练参数量: {trainable_params:,}")

        # 模型移动到设备
        self.model.to(config.device)

        # 创建优化器和调度器
        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, config)

        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode="min",
        )

        # 训练状态
        self.current_epoch = 1
        self.global_step = 0

        # 训练历史
        self.history = {
            "train_loss": [],
            "train_ppl": [],
            "val_loss": [],
            "val_ppl": [],
        }

    def train_epoch(self) -> tuple[float, float]:
        """
        训练一个 epoch

        Returns:
            tuple[float, float]: (平均损失, 平均困惑度)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {self.current_epoch}")

        for batch in pbar:
            # 获取输入和目标
            input_ids = batch["input_ids"].to(self.config.device)
            target_ids = batch["target_ids"].to(self.config.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.config.device)

            # 前向传播
            logits = self.model(input_ids, attention_mask)

            # 计算损失
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            grad_norm = clip_gradients(self.model, self.config.grad_clip)

            # 更新参数
            self.optimizer.step()

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # 记录指标
            total_loss += loss.item()
            self.global_step += 1

            # 计算困惑度
            current_lr = self.optimizer.param_groups[0]["lr"]
            ppl = torch.exp(loss).item()

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{ppl:.4f}",
                    "lr": f"{current_lr:.6f}",
                    "grad_norm": f"{grad_norm:.4f}",
                }
            )

            # 记录到 logger
            self.logger.log_metrics(
                step=self.global_step,
                metrics={
                    "loss": loss.item(),
                    "ppl": ppl,
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                },
                prefix="train",
            )

        avg_loss = total_loss / num_batches
        avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
        return avg_loss, avg_ppl

    def validate_epoch(self) -> tuple[float, float]:
        """
        验证一个 epoch

        Returns:
            tuple[float, float]: (平均损失, 平均困惑度)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")

        with torch.no_grad():
            for batch in pbar:
                # 获取输入和目标
                input_ids = batch["input_ids"].to(self.config.device)
                target_ids = batch["target_ids"].to(self.config.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.config.device)

                # 前向传播
                logits = self.model(input_ids, attention_mask)

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
        return avg_loss, avg_ppl

    def save_checkpoint(self, checkpoint_path: Path, **kwargs) -> None:
        """
        保存检查点

        Args:
            checkpoint_path: 检查点保存路径
            **kwargs: 其他需要保存的信息
        """
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            loss=self.history["val_loss"][-1] if self.history["val_loss"] else 0.0,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
        """
        self.logger.info(f"从检查点恢复训练: {checkpoint_path}")
        checkpoint = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.config.device,
        )
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("step", 0)

    def train(self, resume_from: Optional[Path] = None) -> Dict:
        """
        完整训练流程

        Args:
            resume_from: 恢复训练的检查点路径

        Returns:
            Dict: 训练历史
        """
        # 恢复训练
        if resume_from is not None and resume_from.exists():
            self.load_checkpoint(resume_from)

        self.logger.info("开始训练")
        self.logger.info("=" * 80)

        # 训练循环
        for epoch in range(self.current_epoch, self.config.max_epochs + 1):
            self.current_epoch = epoch

            # 训练一个 epoch
            train_loss, train_ppl = self.train_epoch()

            # 验证一个 epoch
            val_loss, val_ppl = self.validate_epoch()

            # 记录 epoch 信息
            self.logger.log_epoch(
                epoch=epoch,
                train_metrics={"loss": train_loss, "ppl": train_ppl},
                val_metrics={"loss": val_loss, "ppl": val_ppl},
            )

            # 保存历史
            self.history["train_loss"].append(train_loss)
            self.history["train_ppl"].append(train_ppl)
            self.history["val_loss"].append(val_loss)
            self.history["val_ppl"].append(val_ppl)

            # 早停检查
            is_improved = self.early_stopping(
                val_loss=val_loss,
                train_loss=train_loss,
                epoch=epoch,
            )

            # 保存最佳模型
            if is_improved:
                self.save_checkpoint(paths.BEST_MODEL_PATH)
                self.logger.info(f"保存最佳模型: {paths.BEST_MODEL_PATH}")

            # 保存最新模型
            self.save_checkpoint(paths.LAST_MODEL_PATH)

            # 检查早停
            if self.early_stopping.should_stop:
                self.logger.info(f"触发早停: {self.early_stopping.stop_reason}")
                break

        self.logger.info("=" * 80)
        self.logger.info("训练完成")
        self.logger.close()

        return self.history
