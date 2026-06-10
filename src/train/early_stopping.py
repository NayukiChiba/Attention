"""
src/train/early_stopping.py

早停机制
"""

from typing import Optional


class EarlyStopping:
    """
    早停机制

    支持多种停止条件：
    1. 验证集指标无改善
    2. 过拟合检测（train/val差距过大）
    3. 验证集loss波动过小（收敛）

    使用方式:
        early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.001,
            overfitting_threshold=0.5,
        )

        for epoch in range(max_epochs):
            train_loss, val_loss = train_and_validate()

            is_improved = early_stopping(
                val_loss=val_loss,
                train_loss=train_loss,
                epoch=epoch,
            )

            if is_improved:
                save_checkpoint(...)

            if early_stopping.should_stop:
                print(f"触发早停: {early_stopping.stop_reason}")
                break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        overfitting_threshold: Optional[float] = None,
        convergence_window: int = 3,
        convergence_threshold: float = 1e-4,
    ):
        """
        初始化早停机制

        Args:
            patience (int): 容忍轮数，连续无改善则停止
            min_delta (float): 最小改善阈值
            mode (str): 优化模式，"min" 表示越小越好（如 loss），"max" 表示越大越好（如 accuracy）
            overfitting_threshold (Optional[float]): 过拟合阈值，train/val差距超过此值触发早停
                例如：0.5 表示 val_loss - train_loss > 0.5 时停止
            convergence_window (int): 收敛检测窗口大小
            convergence_threshold (float): 收敛阈值，窗口内变化小于此值认为收敛
        """
        # 基本配置
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        # 过拟合检测
        self.overfitting_threshold = overfitting_threshold

        # 收敛检测
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold

        # 状态变量
        self.counter = 0  # 无改善计数器
        self.best_score = None  # 最佳验证指标
        self.best_epoch = 0  # 最佳轮数
        self.should_stop = False  # 是否应该停止
        self.stop_reason = ""  # 停止原因

        # 历史记录（用于收敛检测）
        self.val_history = []

    def __call__(
        self,
        val_loss: float,
        train_loss: Optional[float] = None,
        epoch: Optional[int] = None,
    ) -> bool:
        """
        更新早停状态

        Args:
            val_loss (float): 当前验证集 loss
            train_loss (Optional[float]): 当前训练集 loss（用于过拟合检测）
            epoch (Optional[int]): 当前轮数

        Returns:
            bool: 是否有改善
        """
        # 记录验证集历史
        self.val_history.append(val_loss)

        # 第一次调用，初始化最佳 loss
        if self.best_score is None:
            self.best_score = val_loss
            if epoch is not None:
                self.best_epoch = epoch
            return True

        # 1. 检查是否有改善
        is_improved = self._is_improved(val_loss)

        if is_improved:
            # 有改善：重置计数器，更新最佳 loss
            self.best_score = val_loss
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
        else:
            # 无改善：增加计数器
            self.counter += 1

        # 2. 检查是否触发早停
        self._check_early_stop(val_loss, train_loss)

        return is_improved

    def _is_improved(self, loss: float) -> bool:
        """
        检查是否有改善

        Args:
            loss (float): 当前 loss

        Returns:
            bool: 是否改善
        """
        if self.mode == "min":
            # loss 越小越好
            return loss < self.best_score - self.min_delta
        else:
            # accuracy 越大越好（mode="max"时）
            return loss > self.best_score + self.min_delta

    def _check_early_stop(
        self,
        val_loss: float,
        train_loss: Optional[float] = None,
    ) -> None:
        """
        检查是否触发早停

        检查顺序：
        1. 过拟合检测
        2. 收敛检测
        3. 无改善计数

        Args:
            val_loss (float): 当前验证集 loss
            train_loss (Optional[float]): 当前训练集 loss
        """
        # 1. 过拟合检测
        if self.overfitting_threshold is not None and train_loss is not None:
            if self._check_overfitting(train_loss, val_loss):
                self.should_stop = True
                return

        # 2. 收敛检测
        if self._check_convergence():
            self.should_stop = True
            return

        # 3. 无改善计数
        if self.counter >= self.patience:
            self.should_stop = True
            self.stop_reason = f"验证集 loss 连续 {self.patience} 轮无改善"
            return

    def _check_overfitting(self, train_loss: float, val_loss: float) -> bool:
        """
        检查是否过拟合

        Args:
            train_loss (float): 训练集 loss
            val_loss (float): 验证集 loss

        Returns:
            bool: 是否过拟合
        """
        if self.mode == "min":
            # loss 模式：val_loss - train_loss > threshold
            gap = val_loss - train_loss
        else:
            # accuracy 模式：train_acc - val_acc > threshold
            gap = train_loss - val_loss

        if gap > self.overfitting_threshold:
            self.stop_reason = f"过拟合：train/val 差距 {gap:.4f} 超过阈值 {self.overfitting_threshold:.4f}"
            return True

        return False

    def _check_convergence(self) -> bool:
        """
        检查是否收敛（验证集 loss 波动过小）

        Returns:
            bool: 是否收敛
        """
        # 需要足够的历史数据
        if len(self.val_history) < self.convergence_window:
            return False

        # 取最近 N 个 epoch 的验证集 loss
        recent_losses = self.val_history[-self.convergence_window :]

        # 计算标准差
        mean_loss = sum(recent_losses) / len(recent_losses)
        variance = sum((loss - mean_loss) ** 2 for loss in recent_losses) / len(
            recent_losses
        )
        std = variance**0.5

        # 如果标准差小于阈值，认为已收敛
        if std < self.convergence_threshold:
            self.stop_reason = (
                f"收敛：最近 {self.convergence_window} 轮验证集 loss 标准差 "
                f"{std:.6f} 小于阈值 {self.convergence_threshold:.6f}"
            )
            return True

        return False

    def reset(self) -> None:
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False
        self.stop_reason = ""
        self.val_history = []

    def get_state(self) -> dict:
        """
        获取当前状态

        Returns:
            dict: 状态字典
        """
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("测试早停机制")
    print("=" * 60)

    # 1. 测试基本早停（无改善）
    print("\n1. 测试基本早停（无改善）")
    print("-" * 60)

    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    val_losses = [1.0, 0.9, 0.85, 0.84, 0.83, 0.83, 0.83, 0.82]

    for epoch, val_loss in enumerate(val_losses, 1):
        is_improved = early_stopping(val_loss, epoch=epoch)

        print(
            f"Epoch {epoch}: val_loss={val_loss:.2f}, "
            f"improved={is_improved}, "
            f"counter={early_stopping.counter}/{early_stopping.patience}"
        )

        if early_stopping.should_stop:
            print(f"触发早停: {early_stopping.stop_reason}")
            break

    # 2. 测试过拟合检测
    print("\n2. 测试过拟合检测")
    print("-" * 60)

    early_stopping = EarlyStopping(
        patience=5,
        min_delta=0.0,
        mode="min",
        overfitting_threshold=0.5,  # train/val差距超过0.5触发
    )

    train_losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    val_losses = [1.0, 0.9, 0.85, 0.9, 1.0, 1.1]

    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
        is_improved = early_stopping(val_loss, train_loss, epoch=epoch)

        gap = val_loss - train_loss
        print(
            f"Epoch {epoch}: train={train_loss:.2f}, val={val_loss:.2f}, "
            f"gap={gap:.2f}, improved={is_improved}"
        )

        if early_stopping.should_stop:
            print(f"触发早停: {early_stopping.stop_reason}")
            break

    # 3. 测试收敛检测
    print("\n3. 测试收敛检测")
    print("-" * 60)

    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.0,
        mode="min",
        convergence_window=3,
        convergence_threshold=1e-3,
    )

    val_losses = [1.0, 0.9, 0.85, 0.5, 0.501, 0.5005, 0.5003, 0.5001]

    for epoch, val_loss in enumerate(val_losses, 1):
        is_improved = early_stopping(val_loss, epoch=epoch)

        if len(early_stopping.val_history) >= 3:
            recent = early_stopping.val_history[-3:]
            mean = sum(recent) / len(recent)
            std = (sum((s - mean) ** 2 for s in recent) / len(recent)) ** 0.5
            print(
                f"Epoch {epoch}: val_loss={val_loss:.4f}, "
                f"improved={is_improved}, std={std:.6f}"
            )
        else:
            print(f"Epoch {epoch}: val_loss={val_loss:.4f}, improved={is_improved}")

        if early_stopping.should_stop:
            print(f"触发早停: {early_stopping.stop_reason}")
            break

    # 4. 测试综合场景
    print("\n4. 测试综合场景")
    print("-" * 60)

    early_stopping = EarlyStopping(
        patience=3,
        min_delta=0.01,
        mode="min",
        overfitting_threshold=0.3,
        convergence_window=3,
        convergence_threshold=1e-3,
    )

    train_losses = [1.0, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5]
    val_losses = [1.0, 0.85, 0.8, 0.78, 0.77, 0.76, 0.75]

    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
        is_improved = early_stopping(val_loss, train_loss, epoch=epoch)

        print(
            f"Epoch {epoch}: train={train_loss:.2f}, val={val_loss:.2f}, "
            f"improved={is_improved}, counter={early_stopping.counter}"
        )

        if early_stopping.should_stop:
            print(f"触发早停: {early_stopping.stop_reason}")
            break

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
