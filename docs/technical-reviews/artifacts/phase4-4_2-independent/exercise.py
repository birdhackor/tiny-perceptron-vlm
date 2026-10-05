import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
correction = 0 * x
y = x + correction
y.sum().backward()
print("結果", y.detach())
print("對輸入的敏感度", x.grad)
assert torch.equal(x.grad, torch.tensor([1.0, 1.0]))
