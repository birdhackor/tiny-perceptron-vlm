import torch
from torch import nn

layer = nn.Linear(2, 1)
for parameter in layer.parameters():
    parameter.requires_grad_(False)
layer.eval()
x = torch.ones(1, 2, requires_grad=True)
output = layer(x)
print("輸出可求導", output.requires_grad)
output.sum().backward()
print("固定權重w1與w2", layer.weight)
print("輸入梯度", x.grad)
assert torch.allclose(x.grad, layer.weight)
