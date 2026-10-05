import torch
from torch.nn import functional as F

scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
targets = torch.arange(2)
loss = F.cross_entropy(scores / 0.5, targets)
loss.backward()
print("代價", round(loss.item(), 4))
print("梯度", scores.grad)
