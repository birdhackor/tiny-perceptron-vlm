import torch
from tiny_perceptron.data import render_chat
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss

torch.manual_seed(42)
x, y = render_chat([{"role": "user", "content": "Z"}, {"role": "assistant", "content": "A"}])
model = TinyLM(ModelConfig(width=8))
logits = model(x[None])["logits"]
logits.retain_grad()
masked_loss(logits, y[None]).backward()
print("問題位置logits梯度", logits.grad[0, 2].norm().item())
print("問題字向量梯度", model.embedding.weight.grad[x[2]].norm().item())
