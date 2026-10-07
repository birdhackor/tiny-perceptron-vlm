import torch
from torch import nn
from tiny_perceptron.quantization import QuantizedLinear
layer = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[0.7, 0.2], [-0.7, 1.2]]))
compressed = QuantizedLinear(layer, bits=4)
x = torch.tensor([[1.0, 1.0]])
print("儲存碼型別", compressed.values.dtype, "輸出型別", compressed(x).dtype)
print("原輸出", layer(x).detach().round(decimals=4).tolist())
print("壓縮輸出", compressed(x).round(decimals=4).tolist())
print("壓縮buffer bytes", compressed.storage_bytes())
