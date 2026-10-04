import torch
from tiny_perceptron.model import Block, ModelConfig

torch.manual_seed(42)
config = ModelConfig(width=8)
block = Block(config)
x = torch.randn(1, 3, 8)
y, cache, auxiliary = block(x)
print("輸入", x.shape, "輸出", y.shape)
print("額外代價", auxiliary.item())
assert y.shape == x.shape
