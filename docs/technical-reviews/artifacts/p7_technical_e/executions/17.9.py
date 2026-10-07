import copy
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.quantization import replace_linear_layers
torch.manual_seed(0)
original = TinyLM(ModelConfig(width=8, tied=False)).eval()
ids = torch.tensor([[1, 2, 3]])
with torch.no_grad():
    before = original(ids)["logits"]
    quantized = replace_linear_layers(copy.deepcopy(original), bits=4)
    after = quantized(ids)["logits"]
    original_again = original(ids)["logits"]
print("兩版形狀", tuple(before.shape), tuple(after.shape))
print("原版是否被改", (before - original_again).abs().max().item())
print("量化分數MAE", (before - after).abs().mean().item())
