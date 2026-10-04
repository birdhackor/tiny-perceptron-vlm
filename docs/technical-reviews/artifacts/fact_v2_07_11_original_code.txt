import torch
from tiny_perceptron.data import toy_conversations, render_chat, pad_batch
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss

torch.manual_seed(42)
examples = [render_chat(messages) for messages in toy_conversations()[:2]]
x, y, valid = pad_batch(examples)
model = TinyLM(ModelConfig(width=8))
loss = masked_loss(model(x, valid=valid)["logits"], y)
loss.backward()
print("batch形狀", x.shape, "有效答案", (y != -100).sum().item())
print("問答接口代價", loss.item())
