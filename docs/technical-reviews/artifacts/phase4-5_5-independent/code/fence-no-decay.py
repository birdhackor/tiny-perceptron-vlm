import torch
from torch import nn

w = nn.Parameter(torch.tensor([2.0]))
optimizer = torch.optim.AdamW([w], lr=0.1, weight_decay=0.0)
w.grad = torch.zeros_like(w)
optimizer.step()
print(w.detach())
assert torch.allclose(w.detach(), torch.tensor([2.0]))
