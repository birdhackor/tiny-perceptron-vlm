import torch
from torch import nn
from tiny_perceptron.quantization import QuantizedLinear
layer = nn.Linear(64, 32)
compressed = QuantizedLinear(layer, bits=4)
x = torch.ones(2, 64)
float_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
print("原層數字bytes", float_bytes)
print("壓縮buffer bytes", compressed.storage_bytes())
print("整數碼bytes", compressed.values.numel() * compressed.values.element_size())
print("運算結果", tuple(compressed(x).shape), compressed(x).dtype)
