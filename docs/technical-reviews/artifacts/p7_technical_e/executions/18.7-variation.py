import torch
from torch import nn
torch.manual_seed(0);teacher=nn.Linear(2,3).eval().requires_grad_(False);probe=torch.tensor([[1.,2.]],requires_grad=True)
with torch.no_grad():out=teacher(probe)
print('with no_grad requires_grad',out.requires_grad)
