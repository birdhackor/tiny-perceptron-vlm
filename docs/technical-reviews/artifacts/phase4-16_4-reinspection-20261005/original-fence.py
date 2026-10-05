import torch
from tiny_perceptron.model import TinyLM, ModelConfig

ids = torch.tensor([[1, 2, 3]])
with torch.no_grad():
    for kv_heads in (4, 1):
        model = TinyLM(ModelConfig(width=16, heads=4, kv_heads=kv_heads)).eval()
        cache = model(ids)["cache"]
        size = sum(t.numel() * t.element_size() for pair in cache for t in pair)
        print("KV頭數", kv_heads, "K形狀", tuple(cache[0][0].shape), "bytes", size)
