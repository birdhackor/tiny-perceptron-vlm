import torch
from torch import nn

embedding = nn.Embedding(4, 2)
with torch.no_grad():
    embedding.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
x = embedding(ids)
context = x.reshape(2, -1)
print(x.shape)
print(context)
assert context.shape == (2, 6)
