import torch

z = torch.tensor([1.0, 2.0, -1.0])
weights = (z - z.max()).exp()
p = weights / weights.sum()
print(weights)
print(p, p.sum().item())
print((z + 100).softmax(dim=0))
assert torch.allclose(p, z.softmax(dim=0))
