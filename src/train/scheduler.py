"""
src/train/scheduler.py

学习率调度器创建函数
"""

import math

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LambdaLR,
    StepLR,
)

from config.defaults import TrainingConfig


def createScheduler(
    optimizer: optim.Optimizer, training_config: TrainingConfig
) -> LambdaLR | CosineAnnealingLR | StepLR | ExponentialLR | None:
    """
    根据配置创建学习率调度器

    Args:
        optimizer (optim.Optimizer): 优化器实例
        training_config (TrainingConfig): 训练配置对象

    Returns:
        调度器实例,如果是 constant 则返回 None

    支持的调度器类型:
        - cosine_warmup: 带预热的余弦退火调度器(自定义)
        - cosine: 余弦退火调度器(PyTorch 内置)
        - step: 阶梯式衰减调度器
        - exponential: 指数衰减调度器
        - constant: 恒定学习率(不使用调度器)
    """
    if training_config.scheduler_type == "constant":
        # 恒定学习率,不使用调度器
        return None

    elif training_config.scheduler_type == "cosine":
        # 余弦退火调度器(PyTorch 内置,无 warmup)
        # 学习率从 initial_lr 降到 eta_min
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=training_config.total_steps,  # 周期长度
            eta_min=training_config.learning_rate
            * training_config.min_lr_ratio,  # 最小学习率
        )
        return scheduler

    elif training_config.scheduler_type == "step":
        # 阶梯式衰减调度器
        # 每隔 step_size 步,学习率乘以 gamma
        step_size = training_config.total_steps // 10  # 每 10% 的步数衰减一次
        gamma = 0.8  # 每次衰减到 80%

        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        return scheduler

    elif training_config.scheduler_type == "exponential":
        # 指数衰减调度器
        # 每步学习率乘以 gamma
        # gamma = (min_lr / initial_lr) ^ (1 / total_steps)
        gamma = math.pow(
            training_config.min_lr_ratio, 1.0 / training_config.total_steps
        )

        scheduler = ExponentialLR(
            optimizer,
            gamma=gamma,
        )
        return scheduler
    elif training_config.scheduler_type == "cosine_warmup":
        # 带预热的余弦退火调度器(自定义)
        def lr_lambda(current_step: int) -> float:
            """
            计算学习率缩放因子

            Warmup 阶段: lr = initial_lr × (current_step / warmup_steps)
            Cosine 阶段: lr = min_lr + (initial_lr - min_lr) × 0.5 × (1 + cos(π × progress))

            Args:
                current_step (int): 当前训练步数

            Returns:
                float: 学习率缩放因子(相对于初始学习率)
            """
            # 1. Warmup 阶段:线性增长
            if current_step < training_config.warmup_steps:
                return float(current_step) / float(max(1, training_config.warmup_steps))

            # 2. Cosine 退火阶段
            progress = float(current_step - training_config.warmup_steps) / float(
                max(1, training_config.total_steps - training_config.warmup_steps)
            )

            # Cosine 退火公式: 0.5 * (1 + cos(π * progress))
            # 从 1.0 平滑降到 min_lr_ratio
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

            # 应用最小学习率比例
            return (
                training_config.min_lr_ratio
                + (1.0 - training_config.min_lr_ratio) * cosine_decay
            )

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler
    else:
        raise ValueError(
            f"不支持的调度器类型: {training_config.scheduler_type},"
            f"请使用 'cosine_warmup', 'cosine', 'step', 'exponential' 或 'constant'"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("测试 Scheduler 创建")
    print("=" * 60)

    import torch.nn as nn

    from config.defaults import TrainingConfig

    # 创建一个简单的模型和优化器用于测试
    model = nn.Linear(10, 10)

    def test_scheduler(scheduler_type: str, config_kwargs: dict = None):
        """测试单个调度器"""
        print(f"\n{scheduler_type.upper()} 调度器")
        print("-" * 60)

        # 创建配置
        config_dict = {
            "scheduler_type": scheduler_type,
            "learning_rate": 1.0,
            "warmup_steps": 100,
            "total_steps": 1000,
            "min_lr_ratio": 0.1,
        }
        if config_kwargs:
            config_dict.update(config_kwargs)

        training_config = TrainingConfig(**config_dict)

        # 创建优化器和调度器
        optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
        scheduler = createScheduler(optimizer, training_config)

        if scheduler is None:
            print("调度器类型: None (constant)")
            return

        print(f"调度器类型: {type(scheduler).__name__}")

        # 记录学习率变化
        steps = [0, 100, 200, 500, 800, 1000]
        lrs = []

        for step in steps:
            # 重置优化器
            optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
            scheduler = createScheduler(optimizer, training_config)

            # 更新到指定步数
            for _ in range(step):
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            print(f"  Step {step:4d}: lr = {current_lr:.6f}")

        # 打印学习率曲线(ASCII 图表)
        print("\n学习率变化曲线(每 50 步采样):")
        optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
        scheduler = createScheduler(optimizer, training_config)

        for i in range(0, 501, 50):
            if i > 0:
                for _ in range(50):
                    scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            bar_length = int(lr * 40)
            bar = "█" * bar_length
            print(f"{i:4d} | {bar} {lr:.4f}")

    # 1. 测试 Cosine Warmup
    test_scheduler("cosine_warmup")

    # 2. 测试 Cosine(无 warmup)
    test_scheduler("cosine")

    # 3. 测试 Step
    test_scheduler("step")

    # 4. 测试 Exponential
    test_scheduler("exponential")

    # 5. 测试 Constant
    test_scheduler("constant")

    # 6. 对比所有调度器
    print("\n" + "=" * 60)
    print("所有调度器对比")
    print("=" * 60)

    import matplotlib

    matplotlib.use("Agg")  # 使用非交互式后端
    import matplotlib.pyplot as plt

    scheduler_types = ["cosine_warmup", "cosine", "step", "exponential"]
    colors = ["blue", "green", "red", "orange"]

    plt.figure(figsize=(12, 6))

    for scheduler_type, color in zip(scheduler_types, colors):
        training_config = TrainingConfig(
            scheduler_type=scheduler_type,
            learning_rate=1.0,
            warmup_steps=100,
            total_steps=1000,
            min_lr_ratio=0.1,
        )

        optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
        scheduler = createScheduler(optimizer, training_config)

        lrs = []
        for step in range(1001):
            lrs.append(optimizer.param_groups[0]["lr"])
            if step < 1000:
                scheduler.step()

        plt.plot(lrs, label=scheduler_type, color=color, linewidth=2)

    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedulers Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图表
    from config import paths

    output_path = paths.FIGURES_DIR / "scheduler_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n学习率对比图已保存到: {output_path}")

    # 7. 测试不支持的调度器类型
    print("\n" + "=" * 60)
    print("测试异常处理")
    print("=" * 60)

    training_config_invalid = TrainingConfig(scheduler_type="cosine_warmup")
    training_config_invalid.scheduler_type = "invalid_scheduler"

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    try:
        createScheduler(optimizer, training_config_invalid)
        print("❌ 应该抛出 ValueError")
    except ValueError as e:
        print(f"正确捕获异常: {e}")
        print("✓ 异常处理测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
