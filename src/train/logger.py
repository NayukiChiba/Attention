"""
src/train/logger.py

训练日志记录器
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    训练日志记录器

    使用方式:
        logger = Logger(name="training", log_dir=paths.LOGS_DIR, tensorboard_dir=paths.TENSORBOARD_DIR)
        logger.start()
        logger.info("开始训练")
        logger.log_metrics(step=100, metrics={"loss": 1.23})
        logger.close()
    """

    def __init__(
        self,
        name: str,
        log_dir: Path,
        tensorboard_dir: Optional[Path] = None,
    ):
        """
        初始化 logger

        Args:
            name (str): logger 名称
            log_dir (Path): 日志文件保存目录
            tensorboard_dir (Optional[Path]): TensorBoard 日志目录
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.tensorboard_dir = Path(tensorboard_dir) if tensorboard_dir else None
        self.logger = None
        self.writer = None
        self.timestamp = None
        self.log_file = None

    def start(self) -> "Logger":
        """启动 logger"""
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 使用时间戳命名日志文件
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{self.name}_{self.timestamp}.log"

        # 创建 logger
        self.logger = logging.getLogger(f"{self.name}_{self.timestamp}")
        self.logger.setLevel(logging.DEBUG)

        # 清除已有的 handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 文件格式器
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 控制台格式器
        console_formatter = logging.Formatter(
            fmt="%(levelname)-8s | %(message)s",
        )

        # 文件处理器（记录所有级别）
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 控制台处理器（只显示 INFO 及以上）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 防止日志传播到根 logger
        self.logger.propagate = False

        # TensorBoard Writer
        if self.tensorboard_dir is not None:
            if TENSORBOARD_AVAILABLE:
                tensorboard_log_dir = self.tensorboard_dir / self.timestamp
                self.writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
                self.info(f"TensorBoard 日志目录: {tensorboard_log_dir}")
            else:
                self.warning("TensorBoard 不可用，请安装: pip install tensorboard")

        self.info(f"日志文件: {self.log_file}")
        return self

    def close(self) -> None:
        """关闭 logger 并清理资源"""
        # 关闭 TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        # 关闭所有 handlers
        if self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

    def info(self, message: str) -> None:
        """记录 INFO 级别日志"""
        if self.logger:
            self.logger.info(message)

    def debug(self, message: str) -> None:
        """记录 DEBUG 级别日志"""
        if self.logger:
            self.logger.debug(message)

    def warning(self, message: str) -> None:
        """记录 WARNING 级别日志"""
        if self.logger:
            self.logger.warning(message)

    def error(self, message: str) -> None:
        """记录 ERROR 级别日志"""
        if self.logger:
            self.logger.error(message)

    def critical(self, message: str) -> None:
        """记录 CRITICAL 级别日志"""
        if self.logger:
            self.logger.critical(message)

    def log_metrics(self, step: int, metrics: dict, prefix: str = "") -> None:
        """
        记录训练指标（同时写入日志和 TensorBoard）

        Args:
            step (int): 当前步数
            metrics (dict): 指标字典
            prefix (str): 前缀（如 "train", "val"）
        """
        # 格式化指标字符串
        metrics_str = " | ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ]
        )

        if prefix:
            message = f"[{prefix}] Step {step:6d} | {metrics_str}"
        else:
            message = f"Step {step:6d} | {metrics_str}"

        # 使用 debug 级别，步级指标仅写入文件，不刷控制台
        self.debug(message)

        # 写入 TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    tag = f"{prefix}/{key}" if prefix else key
                    self.writer.add_scalar(tag, value, step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: Optional[dict] = None,
    ) -> None:
        """
        记录每个 epoch 的训练和验证指标

        Args:
            epoch (int): 当前轮数
            train_metrics (dict): 训练指标
            val_metrics (Optional[dict]): 验证指标
        """
        self.info("=" * 80)
        self.info(f"Epoch {epoch}")
        self.info("-" * 80)

        # 训练指标
        train_str = " | ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in train_metrics.items()
            ]
        )
        self.info(f"Train | {train_str}")

        # 验证指标
        if val_metrics:
            val_str = " | ".join(
                [
                    f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in val_metrics.items()
                ]
            )
            self.info(f"Val   | {val_str}")

        self.info("=" * 80)

        # 写入 TensorBoard（按 epoch 记录）
        if self.writer is not None:
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"epoch/train_{key}", value, epoch)

            if val_metrics:
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"epoch/val_{key}", value, epoch)

    def log_config(self, config) -> None:
        """
        记录配置信息

        Args:
            config: 配置对象（支持 dataclass 或字典）
        """
        self.info("=" * 80)
        self.info("Configuration")
        self.info("-" * 80)

        if hasattr(config, "__dataclass_fields__"):
            # dataclass 对象
            from dataclasses import fields

            for field in fields(config):
                value = getattr(config, field.name)
                self.info(f"  {field.name}: {value}")
        elif isinstance(config, dict):
            # 字典
            for key, value in config.items():
                self.info(f"  {key}: {value}")
        else:
            self.info(str(config))

        self.info("=" * 80)


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Logger")
    print("=" * 60)

    from config import paths

    # 1. 创建并启动 logger
    print("\n1. 创建并启动 Logger")
    print("-" * 60)

    logger = Logger(
        name="test",
        log_dir=paths.LOGS_DIR,
        tensorboard_dir=paths.TENSORBOARD_DIR,
    )
    logger.start()
    print()

    # 2. 测试不同级别的日志
    print("2. 测试不同级别的日志")
    print("-" * 60)

    logger.debug("这是 DEBUG 级别的日志（仅文件可见）")
    logger.info("这是 INFO 级别的日志")
    logger.warning("这是 WARNING 级别的日志")
    logger.error("这是 ERROR 级别的日志")
    print()

    # 3. 测试记录训练指标
    print("3. 测试记录训练指标")
    print("-" * 60)

    for step in range(1, 6):
        metrics = {
            "loss": 2.5 - step * 0.2,
            "lr": 1e-3,
            "grad_norm": 0.5,
        }
        logger.log_metrics(step, metrics, prefix="train")

    print()

    # 4. 测试记录 epoch 信息
    print("4. 测试记录 Epoch 信息")
    print("-" * 60)

    train_metrics = {
        "loss": 1.234,
        "perplexity": 3.456,
    }

    val_metrics = {
        "loss": 1.123,
        "perplexity": 3.321,
    }

    logger.log_epoch(epoch=1, train_metrics=train_metrics, val_metrics=val_metrics)
    print()

    # 5. 测试记录配置
    print("5. 测试记录配置")
    print("-" * 60)

    from config.defaults import GPTConfig

    config = GPTConfig(
        vocab_size=5000,
        context_length=256,
        embedding_dim=384,
    )

    logger.log_config(config)
    print()

    # 6. 关闭 logger
    print("6. 关闭 Logger")
    print("-" * 60)
    logger.close()
    print("Logger 已关闭")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
