import torch
from tiny_perceptron.model import TinyLM, ModelConfig

for seed in [1, 2, 3]:
    torch.manual_seed(seed)
    model = TinyLM(ModelConfig(width=8))
    print("種子", seed, "輸入特徵表初值標準差", model.embedding.weight.std().item())
torch.manual_seed(1)
a = TinyLM(ModelConfig(width=8))
torch.manual_seed(1)
b = TinyLM(ModelConfig(width=8))
print("同種子重建輸入特徵表相同", torch.equal(a.embedding.weight, b.embedding.weight))
