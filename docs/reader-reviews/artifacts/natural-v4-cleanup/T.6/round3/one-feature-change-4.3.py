import torch
from torch import nn

x = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])
layer = nn.LayerNorm(3)
y = layer(x)
print(y)
print("平均", y.mean(dim=-1))
print("變異數", y.var(dim=-1, unbiased=False))
assert torch.allclose(y.mean(dim=-1), torch.zeros(1, 2), atol=1e-6)

first = layer(x)
x[0, 1, 0] += 100
shifted = layer(x)
print(torch.allclose(first[0, 0], shifted[0, 0]))
print(torch.allclose(first[0, 1], shifted[0, 1]))
