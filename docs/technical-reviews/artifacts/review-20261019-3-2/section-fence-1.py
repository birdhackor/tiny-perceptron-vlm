import torch

q = torch.tensor([1.0, 0.0])
keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
scores = q @ keys.T
weights = scores.softmax(dim=0)
print(scores)
print(weights)
