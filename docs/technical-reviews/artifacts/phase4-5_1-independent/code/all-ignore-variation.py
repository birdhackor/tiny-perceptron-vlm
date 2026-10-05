import torch
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss

torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
x = torch.tensor([[1, 2, 3]])
y = torch.tensor([[-100, -100, -100]])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
before = masked_loss(model(x)["logits"], y).item()
for step in range(40):
    optimizer.zero_grad()
    loss = masked_loss(model(x)["logits"], y)
    loss.backward()
    if step == 0:
        print("第一步embedding梯度大小", model.embedding.weight.grad.norm().item())
    optimizer.step()
after = masked_loss(model(x)["logits"], y).item()
print("固定例子代價", before, "→", after)
assert after < before
