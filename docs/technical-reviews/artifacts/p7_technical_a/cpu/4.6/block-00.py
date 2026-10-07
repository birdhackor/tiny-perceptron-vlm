import torch
from tiny_perceptron.model import TinyLM, ModelConfig

torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=20, width=8))
ids = torch.tensor([[1, 2, 3]])
logits = model(ids)["logits"]
print(logits.shape)
print("第一位置的前五個候選分數", logits[0, 0, :5])
print("各位置最高分ID", logits.argmax(dim=-1))
assert logits.shape == (1, 3, 20)
