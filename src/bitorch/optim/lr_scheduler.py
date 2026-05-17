from __future__ import annotations

from bitorch.optim.optimizer import BitOptimizer


class BitStepLR:
    """Decay optimizer learning rates every `step_size` calls to `step()`."""

    def __init__(self, optimizer: BitOptimizer, step_size: int, gamma: float = 0.1) -> None:
        if not isinstance(optimizer, BitOptimizer):
            raise TypeError("optimizer must be a BitOptimizer")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.optimizer = optimizer
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.last_epoch = -1
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def step(self) -> list[float]:
        self.last_epoch += 1
        if self.last_epoch > 0 and self.last_epoch % self.step_size == 0:
            for group in self.optimizer.param_groups:
                group["lr"] *= self.gamma
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        return self.get_last_lr()
