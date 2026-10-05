import torch
from torch import nn

embedding = nn.Embedding(5, 3)
with torch.no_grad():
    embedding.weight.copy_(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
    )
ids = torch.tensor([[1, 2, 1]])
x = embedding(ids)
print(x.shape)
print(x)
assert torch.equal(x[0, 0], x[0, 2])
