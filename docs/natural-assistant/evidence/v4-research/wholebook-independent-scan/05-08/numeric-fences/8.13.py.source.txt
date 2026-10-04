import torch
from torch import nn
from tiny_perceptron.alignment import LoRALinear

torch.manual_seed(0)
layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)
with torch.no_grad():
    layer.b.fill_(1.0)
delta1 = layer.merged_weight() - layer.base.weight
layer.alpha = 4
delta2 = layer.merged_weight() - layer.base.weight
print("加倍誤差", (delta2 - 2 * delta1).abs().max().item())
print("符合兩倍", torch.allclose(delta2, 2 * delta1, atol=1e-6, rtol=0))
