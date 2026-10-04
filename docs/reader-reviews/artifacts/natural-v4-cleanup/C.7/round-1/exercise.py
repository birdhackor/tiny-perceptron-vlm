import torch

logits = torch.tensor([0.0, 0.0], requires_grad=True)
action = 1
reward, baseline = 0.0, 0.5
before = logits.detach().softmax(0)
loss = -(reward - baseline) * logits.log_softmax(0)[action]
loss.backward()
with torch.no_grad():
    updated = logits - 0.1 * logits.grad
    after = updated.softmax(0)
print("梯度", logits.grad.tolist())
print("新分數", updated.tolist())
print("前機率", before.tolist())
print("後機率", [round(p, 6) for p in after.tolist()])
