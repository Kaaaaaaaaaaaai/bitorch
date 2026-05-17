from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from bitorch.nn.parameter import BitParameter


class BitOptimizer:
    """Small optimizer base class with PyTorch-like parameter groups."""

    def __init__(self, params: Iterable[BitParameter], defaults: dict[str, Any]) -> None:
        parameters = list(params)
        if not parameters:
            raise ValueError("optimizer received an empty parameter list")
        if not all(isinstance(parameter, BitParameter) for parameter in parameters):
            raise TypeError("BitOptimizer only accepts BitParameter instances")
        self.defaults = dict(defaults)
        self.param_groups = [{"params": parameters, **self.defaults}]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter.zero_grad(set_to_none=set_to_none)
