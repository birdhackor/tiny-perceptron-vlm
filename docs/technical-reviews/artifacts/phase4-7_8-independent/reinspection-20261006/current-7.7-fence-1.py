import torch
from tiny_perceptron.model import TinyLM, ModelConfig

torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=10, width=8))
base = model(torch.tensor([[1, 2, 3]]))["logits"]
ids = torch.tensor([[0, 0, 1, 2, 3]])
valid = ids != 0
positions = torch.tensor([[0, 0, 0, 1, 2]])
actual = model(ids, valid=valid, positions=positions)["logits"][:, 2:]
print("有效位置最大差", (base - actual).abs().max().item())
assert torch.allclose(base, actual, atol=1e-6, rtol=0.0)
