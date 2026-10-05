import torch

w = torch.tensor([1.0, 2.0], requires_grad=True)
first_cost = (w[0] - 3).square()
second_cost = 2 * w[1].square()
loss = first_cost + second_cost
loss.backward()
print("參數", w.detach())
print("各項與總代價", first_cost.item(), second_cost.item(), loss.item())
print("梯度", w.grad)
assert torch.allclose(w.grad, torch.tensor([-4.0, 8.0]))
