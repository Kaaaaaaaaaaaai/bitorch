from __future__ import annotations

import torch

from bitorch.nn.parameter import BitParameter
from bitorch.optim.optimizer import BitOptimizer


class BitSGD(BitOptimizer):
    """SGD over latent float states, then thresholded back to boolean parameters."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(params, defaults={"lr": float(lr), "weight_decay": float(weight_decay)})

    def step(self) -> None:
        for group in self.param_groups:
            learning_rate = group["lr"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if not isinstance(parameter, BitParameter) or parameter.grad is None:
                    continue
                gradient = parameter.grad.to(dtype=torch.float32, device=parameter.device)
                if weight_decay:
                    gradient = gradient + (weight_decay * parameter.latent)
                parameter.latent.add_(gradient, alpha=-learning_rate)
                parameter.sync_from_latent()
