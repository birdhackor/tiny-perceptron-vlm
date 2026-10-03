import torch
from torch import nn
from torch.nn import functional as F

a = nn.Linear(1, 2, bias=False)
b = nn.Linear(2, 1, bias=False)
with torch.no_grad():
    a.weight.copy_(torch.tensor([[1.0], [-1.0]]))
    b.weight.copy_(torch.tensor([[1.0, 1.0]]))
x = torch.tensor([[-2.0], [0.0], [3.0]])
hidden = a(x)
linear = b(hidden)
curved = b(hidden)
combined = x @ (b.weight @ a.weight).T
print(hidden)
print("純線性", linear.squeeze(-1))
print("加ReLU", curved.squeeze(-1))
assert torch.allclose(linear, combined)
