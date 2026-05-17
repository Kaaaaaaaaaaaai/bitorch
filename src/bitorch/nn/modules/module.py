from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator

import torch

from bitorch.nn.parameter import BitParameter


class BitModule(torch.nn.Module):
    """Minimal PyTorch-like module base class for BitParameters."""

    def __init__(self) -> None:
        super().__init__()
        self._bit_parameters: OrderedDict[str, BitParameter] = OrderedDict()

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        parameters = self.__dict__.get("_bit_parameters")
        if parameters is None:
            return
        if isinstance(value, BitParameter):
            parameters[name] = value
        elif name in parameters:
            parameters.pop(name, None)

    def register_bit_parameter(self, name: str, parameter: BitParameter | None) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("parameter name must be a non-empty string")
        if parameter is not None and not isinstance(parameter, BitParameter):
            raise TypeError("parameter must be a BitParameter or None")
        setattr(self, name, parameter)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
    ) -> Iterator[tuple[str, BitParameter]]:
        for name, parameter in self._bit_parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, parameter

        if not recurse:
            return

        for child_name, child in self.named_children():
            if not isinstance(child, BitModule):
                continue
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            yield from child.named_parameters(prefix=child_prefix, recurse=True)

    def parameters(self, recurse: bool = True) -> Iterator[BitParameter]:
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters():
            parameter.zero_grad(set_to_none=set_to_none)
