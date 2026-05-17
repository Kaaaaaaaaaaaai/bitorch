from __future__ import annotations

from typing import Any

from .tensor import BitTensor


def bit_and(left: Any, right: Any) -> BitTensor:
    return BitTensor.ensure(left) & BitTensor.ensure(right)


def bit_or(left: Any, right: Any) -> BitTensor:
    return BitTensor.ensure(left) | BitTensor.ensure(right)


def bit_xor(left: Any, right: Any) -> BitTensor:
    return BitTensor.ensure(left) ^ BitTensor.ensure(right)


def bit_not(value: Any) -> BitTensor:
    return ~BitTensor.ensure(value)
