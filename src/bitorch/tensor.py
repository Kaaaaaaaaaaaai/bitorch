from collections.abc import Callable, Iterable
from typing import Any

import torch


BackwardResult = Iterable[tuple["BitTensor", torch.Tensor | None]]
BackwardFn = Callable[[torch.Tensor], BackwardResult]


class BitTensor(torch.Tensor):
    """Boolean tensor with a small custom STE-based autograd tape."""

    @staticmethod
    def __new__(
        cls,
        data: Any,
        requires_grad: bool = True,
        *,
        _parents: tuple["BitTensor", ...] = (),
        _backward: BackwardFn | None = None,
        _op: str = "leaf",
    ) -> "BitTensor":
        base = torch.as_tensor(data, dtype=torch.bool)
        result = torch.Tensor._make_subclass(cls, base, False)
        result.grad_dtype = torch.float32
        result._bit_requires_grad = bool(requires_grad)
        result._parents = _parents
        result._backward = _backward
        result._op = _op
        return result

    @classmethod
    def ensure(cls, value: Any) -> "BitTensor":
        if isinstance(value, BitTensor):
            return value
        return cls(value)

    @classmethod
    def ensure_bool_tensor_constant(cls, value: Any) -> "BitTensor":
        if isinstance(value, BitTensor):
            return value
        if isinstance(value, torch.Tensor) and value.dtype is not torch.bool:
            raise TypeError("BitTensor-compatible torch.Tensor operands must have dtype torch.bool")
        return cls(value, requires_grad=False)

    @classmethod
    def zeros(
        cls,
        *shape: int,
        requires_grad: bool = False,
        device: torch.device | str | None = None,
    ) -> "BitTensor":
        return cls(torch.zeros(shape, dtype=torch.bool, device=device), requires_grad=requires_grad)

    @classmethod
    def ones(
        cls,
        *shape: int,
        requires_grad: bool = False,
        device: torch.device | str | None = None,
    ) -> "BitTensor":
        return cls(torch.ones(shape, dtype=torch.bool, device=device), requires_grad=requires_grad)

    @property
    def requires_grad(self) -> bool:
        return self._bit_requires_grad

    def requires_grad_(self, requires_grad: bool = True) -> "BitTensor":
        self._bit_requires_grad = bool(requires_grad)
        return self

    def to_tensor(self) -> torch.Tensor:
        return self.as_subclass(torch.Tensor)

    def detach(self) -> "BitTensor":
        return BitTensor(self.to_tensor(), requires_grad=False)

    def clone(self) -> "BitTensor":
        output = BitTensor(
            self.to_tensor().clone(),
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="clone",
        )

        def _backward(gradient: torch.Tensor) -> BackwardResult:
            if self.requires_grad:
                yield self, gradient

        output._backward = _backward
        return output

    def zero_grad(self, set_to_none: bool = True) -> None:
        if set_to_none:
            self.grad = None
            return
        if self.grad is None:
            self.grad = torch.zeros_like(self.to_tensor(), dtype=torch.float32)
        else:
            self.grad.zero_()

    def backward(self, gradient: torch.Tensor | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a BitTensor that does not require gradients")

        if gradient is None:
            if self.numel() != 1:
                raise RuntimeError("gradient must be supplied for non-scalar BitTensor outputs")
            gradient = torch.ones_like(self.to_tensor(), dtype=torch.float32)
        else:
            gradient = torch.as_tensor(gradient, dtype=torch.float32, device=self.device)
            if tuple(gradient.shape) != tuple(self.shape):
                raise RuntimeError(
                    f"gradient shape {tuple(gradient.shape)} does not match BitTensor shape {tuple(self.shape)}"
                )

        topo: list[BitTensor] = []
        visited: set[int] = set()

        def build(node: BitTensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for parent in node._parents:
                build(parent)
            topo.append(node)

        build(self)

        pending: dict[int, torch.Tensor] = {id(self): gradient}
        for node in reversed(topo):
            node_gradient = pending.get(id(node))
            if node_gradient is None:
                continue

            if not node._parents and node.requires_grad:
                node._accumulate_grad(node_gradient)

            if node._backward is None:
                continue

            for parent, parent_gradient in node._backward(node_gradient):
                if parent_gradient is None or not parent.requires_grad:
                    continue
                parent_gradient = parent_gradient.to(dtype=torch.float32, device=parent.device)
                parent_id = id(parent)
                if parent_id in pending:
                    pending[parent_id] = pending[parent_id] + parent_gradient
                else:
                    pending[parent_id] = parent_gradient

    def _accumulate_grad(self, gradient: torch.Tensor) -> None:
        gradient = gradient.detach().to(dtype=torch.float32, device=self.device)
        if self.grad is None:
            self.grad = gradient.clone()
        else:
            self.grad = self.grad + gradient

    def _binary_tensor_op(
        self,
        other: Any,
        *,
        op_name: str,
        forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        grad_left: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        grad_right: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> "BitTensor":
        right = BitTensor.ensure(other)
        left_bits = self.to_tensor()
        right_bits = right.to_tensor()
        output_bits = forward(left_bits, right_bits)
        requires_grad = self.requires_grad or right.requires_grad
        output = BitTensor(
            output_bits,
            requires_grad=requires_grad,
            _parents=(self, right),
            _op=op_name,
        )

        def _backward(gradient: torch.Tensor) -> BackwardResult:
            left_float = left_bits.to(torch.float32)
            right_float = right_bits.to(torch.float32)
            if self.requires_grad:
                left_gradient = gradient * grad_left(left_float, right_float)
                yield self, _sum_to_shape(left_gradient, tuple(self.shape))
            if right.requires_grad:
                right_gradient = gradient * grad_right(left_float, right_float)
                yield right, _sum_to_shape(right_gradient, tuple(right.shape))

        output._backward = _backward if requires_grad else None
        return output

    def __and__(self, other: Any) -> "BitTensor":
        return self._binary_tensor_op(
            other,
            op_name="and",
            forward=torch.logical_and,
            grad_left=lambda _left, right: right,
            grad_right=lambda left, _right: left,
        )

    def __rand__(self, other: Any) -> "BitTensor":
        return BitTensor.ensure(other).__and__(self)

    def __or__(self, other: Any) -> "BitTensor":
        return self._binary_tensor_op(
            other,
            op_name="or",
            forward=torch.logical_or,
            grad_left=lambda _left, right: 1.0 - right,
            grad_right=lambda left, _right: 1.0 - left,
        )

    def __ror__(self, other: Any) -> "BitTensor":
        return BitTensor.ensure(other).__or__(self)

    def __xor__(self, other: Any) -> "BitTensor":
        return self._binary_tensor_op(
            other,
            op_name="xor",
            forward=torch.logical_xor,
            grad_left=lambda _left, right: 1.0 - (2.0 * right),
            grad_right=lambda left, _right: 1.0 - (2.0 * left),
        )

    def __rxor__(self, other: Any) -> "BitTensor":
        return BitTensor.ensure(other).__xor__(self)

    def __invert__(self) -> "BitTensor":
        bits = torch.logical_not(self.to_tensor())
        output = BitTensor(
            bits,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="not",
        )

        def _backward(gradient: torch.Tensor) -> BackwardResult:
            if self.requires_grad:
                yield self, -gradient

        output._backward = _backward if self.requires_grad else None
        return output

    def logical_and(self, other: Any) -> "BitTensor":
        return self & other

    def logical_or(self, other: Any) -> "BitTensor":
        return self | other

    def logical_xor(self, other: Any) -> "BitTensor":
        return self ^ other

    def logical_not(self) -> "BitTensor":
        return ~self

    def __repr__(self) -> str:
        return f"BitTensor({self.to_tensor().__repr__()}, requires_grad={self.requires_grad})"


def _sum_to_shape(gradient: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    while gradient.ndim > len(shape):
        gradient = gradient.sum(dim=0)

    for dim, size in enumerate(shape):
        if size == 1 and gradient.shape[dim] != 1:
            gradient = gradient.sum(dim=dim, keepdim=True)

    return gradient
