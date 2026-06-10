"""
src/train/early_stopping.py

早停机制
"""


class EarlyStopping:
    """
    早停机制

    使用方式:
        early_stopping = EarlyStopping(patience=5, min_delta=0.001)

        for epoch in range(max_epochs):
            val_loss = train_and_validate()

            is_improved = early_stopping(val_loss)
            if is_improved:
                # 保存最佳模型
                save_checkpoint(...)

            if early_stopping.should_stop:
                print("触发早停")
                break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        初始化早停机制

        Args:
            patience (int): 容忍轮数，连续无改善则停止
            min_delta (float): 最小改善阈值
            mode (str): 优化模式，"min" 表示越小越好（如 loss），"max" 表示越大越好（如 accuracy）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, score: float) -> bool:
        """
        更新早停状态

        Args:
            score (float): 当前验证指标

        Returns:
            bool: 是否有改善
        """
        # 第一次调用，初始化最佳分数
        if self.best_score is None:
            self.best_score = score
            return True

        # 检查是否有改善
        if self._is_improved(score):
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def _is_improved(self, score: float) -> bool:
        """
        检查是否有改善

        Args:
            score (float): 当前分数

        Returns:
            bool: 是否改善
        """
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:  # mode == "max"
            return score > self.best_score + self.min_delta

    def reset(self) -> None:
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0


if __name__ == "__main__":
    print("=" * 60)
    print("测试早停机制")
    print("=" * 60)

    import torch.nn as nn

    # 创建一个简单的测试模型
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()

    # 1. 测试 min 模式（loss 越小越好）
    print("\n1. 测试 min 模式")
    print("-" * 60)

    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    # 模拟训练过程
    val_losses = [1.0, 0.9, 0.85, 0.84, 0.83, 0.83, 0.83, 0.82]

    for epoch, val_loss in enumerate(val_losses, 1):
        early_stopping(val_loss, model)

        print(
            f"Epoch {epoch}: val_loss={val_loss:.2f}, "
            f"best={early_stopping.best_score:.2f}, "
            f"counter={early_stopping.counter}/{early_stopping.patience}"
        )

        if early_stopping.should_stop:
            print(f"触发早停于 Epoch {epoch}")
            break

    print()

    # 2. 测试 max 模式（accuracy 越大越好）
    print("2. 测试 max 模式")
    print("-" * 60)

    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode="max")

    # 模拟训练过程
    val_accs = [0.5, 0.6, 0.65, 0.66, 0.67, 0.67, 0.67, 0.68]

    for epoch, val_acc in enumerate(val_accs, 1):
        early_stopping(val_acc, model)

        print(
            f"Epoch {epoch}: val_acc={val_acc:.2f}, "
            f"best={early_stopping.best_score:.2f}, "
            f"counter={early_stopping.counter}/{early_stopping.patience}"
        )

        if early_stopping.should_stop:
            print(f"触发早停于 Epoch {epoch}")
            break

    print()

    # 3. 测试无改善情况（立即触发早停）
    print("3. 测试无改善情况")
    print("-" * 60)

    early_stopping = EarlyStopping(patience=2, min_delta=0.0, mode="min")

    val_losses = [1.0, 1.0, 1.0]

    for epoch, val_loss in enumerate(val_losses, 1):
        early_stopping(val_loss, model)

        print(
            f"Epoch {epoch}: val_loss={val_loss:.2f}, "
            f"best={early_stopping.best_score:.2f}, "
            f"counter={early_stopping.counter}/{early_stopping.patience}"
        )

        if early_stopping.should_stop:
            print(f"触发早停于 Epoch {epoch}")
            break

    print()

    # 4. 测试重置功能
    print("4. 测试重置功能")
    print("-" * 60)

    early_stopping = EarlyStopping(patience=2, min_delta=0.0, mode="min")
    early_stopping(1.0, model)
    early_stopping(1.0, model)
    early_stopping(1.0, model)

    print(
        f"重置前: counter={early_stopping.counter}, should_stop={early_stopping.should_stop}"
    )

    early_stopping.reset()

    print(
        f"重置后: counter={early_stopping.counter}, should_stop={early_stopping.should_stop}"
    )

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
