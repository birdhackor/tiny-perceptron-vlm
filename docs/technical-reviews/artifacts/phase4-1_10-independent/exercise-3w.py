import torch

w = torch.tensor(3.0, requires_grad=True)
u = 3 * w
loss = u.square()
loss.backward()
print("中間值", u.item(), "代價", loss.item())
print("兩段敏感度", 2 * u.detach().item(), 3)
print("合成梯度", w.grad.item())
assert w.grad.item() == 54
