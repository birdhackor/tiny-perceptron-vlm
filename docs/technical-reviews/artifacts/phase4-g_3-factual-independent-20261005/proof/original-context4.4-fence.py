import torch
from tiny_perceptron.modern import DenseFFN

torch.manual_seed(42)
layer = DenseFFN(4)
x = torch.ones(1, 2, 4)
y = layer(x)
print("擴張權重", layer.up.weight.shape)
print("縮回權重", layer.down.weight.shape)
print(y.shape, y)
assert torch.allclose(y[:, 0], y[:, 1])
