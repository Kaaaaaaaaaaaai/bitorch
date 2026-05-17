from __future__ import annotations

from typing import Any

import torch

from bitorch.tensor import BitTensor


class BitParameter(BitTensor):
    """Trainable boolean parameter backed by a float latent state."""

    @staticmethod
    def __new__(cls, data: Any, requires_grad: bool = True) -> "BitParameter":
        parameter = super(BitParameter, cls).__new__(cls, data, requires_grad=requires_grad)
        parameter._latent = torch.where(
            parameter.to_tensor(),
            torch.ones_like(parameter.to_tensor(), dtype=torch.float32),
            -torch.ones_like(parameter.to_tensor(), dtype=torch.float32),
        )
        return parameter

    @property
    def latent(self) -> torch.Tensor:
        return self._latent

    def sync_from_latent(self) -> None:
        self.to_tensor().copy_(self._latent >= 0.0)
