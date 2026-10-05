import torch
from torch import nn

layer = nn.Linear(3, 2)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[1.0, 0.0, 2.0], [0.0, -1.0, 1.0]]))
    layer.bias.copy_(torch.tensor([1.0, 0.0]))
x = torch.tensor([[1.0, 2.0, 4.0]])
y = layer(x)
manual = x @ layer.weight.T + layer.bias
print(layer.weight.shape, y)
assert torch.allclose(y, manual)
