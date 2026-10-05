import torch

values = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
weights = torch.tensor([0.1, 0.9])
output = weights @ values
print(output)
assert torch.allclose(output, torch.tensor([0.2, 3.6]))
