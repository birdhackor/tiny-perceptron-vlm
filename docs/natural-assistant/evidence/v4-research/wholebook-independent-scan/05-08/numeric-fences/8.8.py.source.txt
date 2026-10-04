import torch
from torch import nn
from tiny_perceptron.alignment import LoRALinear

torch.manual_seed(0)
layer = LoRALinear(nn.Linear(16, 12), rank=2, alpha=2)
x = torch.randn(3, 16)
print("可訓練參數", sum(p.numel() for p in layer.parameters() if p.requires_grad))
print("初始相同", torch.equal(layer(x), layer.base(x)))
layer(x).sum().backward()
print("A梯度為零", layer.a.grad.abs().max().item() == 0)
print("B梯度非零", layer.b.grad.abs().max().item() > 0)
