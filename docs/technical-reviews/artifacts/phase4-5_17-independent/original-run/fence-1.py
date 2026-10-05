import torch
from torch import nn

layer = nn.Linear(2, 1)
x = torch.ones(1, 2)
layer.eval()
output = layer(x)
print("eval輸出可求導", output.requires_grad, output.grad_fn)
output.sum().backward()
with torch.no_grad():
    other = layer(x)
print("no_grad輸出可求導", other.requires_grad, other.grad_fn)
assert output.requires_grad and not other.requires_grad
