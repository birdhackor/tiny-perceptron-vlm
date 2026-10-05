import torch
from torch import nn

w = nn.Parameter(torch.tensor(2.0))
(w.square()).backward()
first = w.grad.clone()
(w.square()).backward()
second = w.grad.clone()
w.grad = None
(w.square()).backward()
print(first.item(), second.item(), w.grad.item(), w.item())
assert second.item() == 2 * first.item()
