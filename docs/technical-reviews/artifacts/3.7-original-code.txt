import torch
from tiny_perceptron.attention import CausalAttention

torch.manual_seed(42)
layer = CausalAttention(width=8, heads=2)
x = torch.randn(1, 4, 8)
out, cache = layer(x)
print("輸出", out.shape)
print("保存的K", cache[0].shape)
assert out.shape == x.shape
