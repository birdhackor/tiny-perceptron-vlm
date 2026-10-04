import torch
from torch import nn

from tiny_perceptron.quantization import QuantizedLinear

torch.manual_seed(42)
layer = nn.Linear(8, 4)
compressed = QuantizedLinear(layer, bits=8)
x = torch.ones(1, 8)
with torch.no_grad():
    error = (layer(x) - compressed(x)).abs().max().item()
print("FP32權重加bias bytes", sum(p.numel() * p.element_size() for p in layer.parameters()))
print("打包權重、刻度與bias bytes", compressed.storage_bytes(), "最大輸出差", error)
