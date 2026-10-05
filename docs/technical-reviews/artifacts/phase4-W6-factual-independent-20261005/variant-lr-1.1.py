import torch

w = torch.tensor(1.0, requires_grad=True)
loss = (w - 3).square()
loss.backward()
print(loss.item(), w.grad.item())
with torch.no_grad():
    w -= 1.1 * w.grad
print(w.item(), (w - 3).square().item())
