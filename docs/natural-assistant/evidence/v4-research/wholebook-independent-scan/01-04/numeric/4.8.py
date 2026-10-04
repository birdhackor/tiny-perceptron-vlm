import torch
from tiny_perceptron.model import TinyLM, ModelConfig

for depth in [1, 2, 3]:
    model = TinyLM(ModelConfig(vocab_size=20, width=8, layers=depth))
    count = model.description()["parameters"]
    logits = model(torch.tensor([[1, 2]]))["logits"]
    print("層數", depth, "參數格數", count, "輸出", logits.shape)
    assert logits.shape == (1, 2, 20)
