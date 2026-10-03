import torch
from torch import nn

e = nn.Embedding(4, 3)
p = nn.Embedding(5, 3)
with torch.no_grad():
    e.weight.copy_(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    p.weight.copy_(torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9], [0.4, 0.8, 1.2]]))
    p.weight.zero_()
ids = torch.tensor([1, 2, 1])
positions = torch.arange(3)
x = e(ids) + p(positions)
print(positions)
print(x)
assert torch.equal(x[0], x[2])
