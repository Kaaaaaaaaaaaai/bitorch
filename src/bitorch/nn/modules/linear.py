import math
from typing import Any

import torch

from bitorch.nn.modules.module import BitModule
from bitorch.nn.parameter import BitParameter
from bitorch.tensor import BackwardResult, BitTensor


class BitLinear(BitModule):
    """Boolean linear layer using AND for multiplication, OR for reduction, XOR for bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = BitParameter(torch.empty((out_features, in_features), dtype=torch.bool, device=device))
        self.bias = BitParameter(torch.empty(out_features, dtype=torch.bool, device=device)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        probability = min(0.5, 1.0 / math.sqrt(self.in_features))
        self.weight.to_tensor().copy_(torch.rand_like(self.weight.to_tensor(), dtype=torch.float32) < probability)
        self.weight._latent.copy_(torch.where(self.weight.to_tensor(), 1.0, -1.0))
        if self.bias is not None:
            self.bias.to_tensor().copy_(torch.zeros_like(self.bias.to_tensor(), dtype=torch.bool))
            self.bias._latent.copy_(torch.where(self.bias.to_tensor(), 1.0, -1.0))

    def forward(self, input: BitTensor | Any) -> BitTensor:
        bit_input = BitTensor.ensure_bool_tensor_constant(input)
        if bit_input.ndim == 0:
            raise ValueError("BitLinear input must have at least one dimension")
        if bit_input.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dimension to be {self.in_features}, got {bit_input.shape[-1]}"
            )

        input_bits = bit_input.to_tensor()
        weight_bits = self.weight.to_tensor()
        weight_view_shape = (1,) * (input_bits.ndim - 1) + (self.out_features, self.in_features)
        expanded_weight = weight_bits.view(weight_view_shape)
        pairwise = torch.logical_and(input_bits.unsqueeze(-2), expanded_weight)
        pre_bias = pairwise.any(dim=-1)

        if self.bias is None:
            output_bits = pre_bias
            bias_bits = None
            expanded_bias = None
        else:
            bias_bits = self.bias.to_tensor()
            bias_view_shape = (1,) * (pre_bias.ndim - 1) + (self.out_features,)
            expanded_bias = bias_bits.view(bias_view_shape)
            output_bits = torch.logical_xor(pre_bias, expanded_bias)

        parents: tuple[BitTensor, ...]
        if self.bias is None:
            parents = (bit_input, self.weight)
        else:
            parents = (bit_input, self.weight, self.bias)

        requires_grad = any(parent.requires_grad for parent in parents)
        output = BitTensor(
            output_bits,
            requires_grad=requires_grad,
            _parents=parents,
            _op="bit_linear",
        )

        def _backward(gradient: torch.Tensor) -> BackwardResult:
            grad_output = gradient.to(torch.float32)
            if bias_bits is None:
                grad_pre_bias = grad_output
            else:
                grad_pre_bias = grad_output * (1.0 - (2.0 * expanded_bias.to(torch.float32)))
                if self.bias is not None and self.bias.requires_grad:
                    grad_bias_full = grad_output * (1.0 - (2.0 * pre_bias.to(torch.float32)))
                    bias_reduce_dims = tuple(range(grad_bias_full.ndim - 1))
                    grad_bias = (
                        grad_bias_full.sum(dim=bias_reduce_dims)
                        if bias_reduce_dims
                        else grad_bias_full
                    )
                    yield self.bias, grad_bias

            active_count = pairwise.sum(dim=-1, keepdim=True)
            pairwise_float = pairwise.to(torch.float32)
            no_active = (active_count == 0).to(torch.float32).expand_as(pairwise_float)
            single_active = (active_count == 1).to(torch.float32).expand_as(pairwise_float)
            or_gradient_mask = no_active + (single_active * pairwise_float)
            grad_pairwise = grad_pre_bias.unsqueeze(-1) * or_gradient_mask

            if bit_input.requires_grad:
                grad_input = (grad_pairwise * expanded_weight.to(torch.float32)).sum(dim=-2)
                yield bit_input, grad_input

            if self.weight.requires_grad:
                expanded_input = input_bits.unsqueeze(-2).to(torch.float32)
                grad_weight_full = grad_pairwise * expanded_input
                weight_reduce_dims = tuple(range(grad_weight_full.ndim - 2))
                grad_weight = (
                    grad_weight_full.sum(dim=weight_reduce_dims)
                    if weight_reduce_dims
                    else grad_weight_full
                )
                yield self.weight, grad_weight

        output._backward = _backward if requires_grad else None
        return output
