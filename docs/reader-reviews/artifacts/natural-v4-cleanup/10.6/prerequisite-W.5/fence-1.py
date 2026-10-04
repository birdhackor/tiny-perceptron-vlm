import torch

recipes = torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0]])
weights = torch.tensor([[10.0, 0.0], [20.0, 1.0], [30.0, 4.0]])
result = recipes @ weights
print(result)
print(result.shape)
