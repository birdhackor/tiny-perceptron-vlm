import torch

q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
scores = q @ k.T
print("Q", q.shape, "K轉置", k.T.shape)
print(scores)
assert scores.shape == (2, 3)
