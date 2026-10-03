import torch

scores = torch.tensor([[60.0, 70.0, 80.0], [80.0, 90.0, 100.0]])
print(scores.shape)
print(scores[0])
print(scores[1, 2].item())
print(scores.mean(dim=1))
print(scores.mean(dim=0))
