`bitorch` is a compact boolean tensor library with a PyTorch-shaped API.

Implemented surface:

- `BitTensor`: `torch.Tensor` subclass backed by `torch.bool`
- Custom STE-style backward traversal with float-valued `.grad`
- Boolean tensor operators: `AND`, `OR`, `XOR`, `NOT`
- `BitParameter`, `BitModule`, and `BitLinear`
- `BitSGD` optimizer over latent float parameter states
- `BitStepLR` scheduler

`BitLinear` uses boolean algebra rather than arithmetic:

- pairwise multiply -> `AND`
- reduction sum -> `OR`
- bias combine -> `XOR`

```python
import torch

from bitorch import BitTensor
from bitorch.nn import BitLinear
from bitorch.optim import BitSGD

layer = BitLinear(3, 2)
inputs = BitTensor([[True, False, True]], requires_grad=True)

outputs = layer(inputs)
outputs.backward(torch.ones_like(outputs.to_tensor(), dtype=torch.float32))

optimizer = BitSGD(layer.parameters(), lr=0.1)
optimizer.step()
optimizer.zero_grad()
```
