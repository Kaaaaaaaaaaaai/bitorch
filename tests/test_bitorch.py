import unittest

import torch

from bitorch import BitTensor
from bitorch.nn import BitLinear, BitParameter
from bitorch.optim import BitSGD, BitStepLR


class BitTensorTests(unittest.TestCase):
    def test_and_forward_and_backward(self) -> None:
        left = BitTensor([True, False], requires_grad=True)
        right = BitTensor([True, True], requires_grad=True)

        output = left & right
        self.assertTrue(torch.equal(output.to_tensor(), torch.tensor([True, False])))

        output.backward(torch.ones(2))
        self.assertTrue(torch.equal(left.grad, torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.equal(right.grad, torch.tensor([1.0, 0.0])))

    def test_xor_backward_uses_surrogate_sign(self) -> None:
        left = BitTensor([True, False], requires_grad=True)
        right = BitTensor([False, True], requires_grad=True)

        output = left ^ right
        output.backward(torch.ones(2))

        self.assertTrue(torch.equal(left.grad, torch.tensor([1.0, -1.0])))
        self.assertTrue(torch.equal(right.grad, torch.tensor([-1.0, 1.0])))


class BitLinearTests(unittest.TestCase):
    def test_forward_matches_boolean_linear_algebra(self) -> None:
        layer = BitLinear(3, 2, bias=True)
        layer.weight = BitParameter(
            [
                [True, False, True],
                [False, True, False],
            ]
        )
        layer.bias = BitParameter([False, True])

        inputs = BitTensor(
            [
                [True, False, True],
                [False, True, False],
            ],
            requires_grad=True,
        )
        output = layer(inputs)

        expected = torch.tensor(
            [
                [True, True],
                [False, False],
            ]
        )
        self.assertTrue(torch.equal(output.to_tensor(), expected))

    def test_forward_accepts_bool_tensor_input(self) -> None:
        layer = BitLinear(3, 2, bias=True)
        layer.weight = BitParameter(
            [
                [True, False, True],
                [False, True, False],
            ]
        )
        layer.bias = BitParameter([False, True])

        inputs = torch.tensor(
            [
                [True, False, True],
                [False, True, False],
            ],
            dtype=torch.bool,
        )
        output = layer(inputs)

        expected = torch.tensor(
            [
                [True, True],
                [False, False],
            ]
        )
        self.assertIsInstance(output, BitTensor)
        self.assertTrue(torch.equal(output.to_tensor(), expected))

    def test_bool_tensor_input_is_constant_for_backward(self) -> None:
        layer = BitLinear(3, 2, bias=True)
        layer.weight = BitParameter(
            [
                [True, False, True],
                [False, True, False],
            ]
        )
        layer.bias = BitParameter([False, True])
        inputs = torch.tensor([[True, False, True]], dtype=torch.bool)

        output = layer(inputs)
        output.backward(torch.ones_like(output.to_tensor(), dtype=torch.float32))

        self.assertIsNone(inputs.grad)
        self.assertEqual(tuple(layer.weight.grad.shape), (2, 3))
        self.assertEqual(tuple(layer.bias.grad.shape), (2,))

    def test_forward_rejects_non_bool_tensor_input(self) -> None:
        layer = BitLinear(3, 2)

        with self.assertRaisesRegex(TypeError, "dtype torch.bool"):
            layer(torch.ones(1, 3))

    def test_backward_populates_input_weight_and_bias_grads(self) -> None:
        layer = BitLinear(3, 2, bias=True)
        layer.weight = BitParameter(
            [
                [True, False, True],
                [False, True, False],
            ]
        )
        layer.bias = BitParameter([False, True])

        inputs = BitTensor(
            [
                [True, False, True],
                [False, True, False],
            ],
            requires_grad=True,
        )
        output = layer(inputs)
        output.backward(torch.ones_like(output.to_tensor(), dtype=torch.float32))

        self.assertEqual(tuple(inputs.grad.shape), (2, 3))
        self.assertEqual(tuple(layer.weight.grad.shape), (2, 3))
        self.assertEqual(tuple(layer.bias.grad.shape), (2,))


class OptimizerTests(unittest.TestCase):
    def test_sgd_updates_latent_state_and_bits(self) -> None:
        parameter = BitParameter([True, False])
        parameter.grad = torch.tensor([2.0, -2.0])
        optimizer = BitSGD([parameter], lr=1.0)

        optimizer.step()

        self.assertTrue(torch.equal(parameter.to_tensor(), torch.tensor([False, True])))

    def test_step_lr_decays_learning_rate(self) -> None:
        parameter = BitParameter([True])
        optimizer = BitSGD([parameter], lr=1.0)
        scheduler = BitStepLR(optimizer, step_size=2, gamma=0.5)

        self.assertEqual(scheduler.step(), [1.0])
        self.assertEqual(scheduler.step(), [1.0])
        self.assertEqual(scheduler.step(), [0.5])


if __name__ == "__main__":
    unittest.main()
