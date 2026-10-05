import torch

w = torch.tensor(1.0, requires_grad=True)
before = (w - 3).square()
before.backward()
print("更新前", w.item(), before.item(), w.grad.item())
with torch.no_grad():
    w -= 0.0 * w.grad
after = (w - 3).square()
print("更新後", w.item(), after.item())
assert after.item() < before.item()
