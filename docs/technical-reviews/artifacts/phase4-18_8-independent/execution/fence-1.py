import torch
from tiny_perceptron.alignment import distillation_kl

student = torch.zeros(1, 3, 2, requires_grad=True)
p = [0.8, 0.2]
teacher = torch.tensor([[p, p, p]]).log().requires_grad_()
labels = torch.tensor([[-100, 0, 1]])
loss = distillation_kl(student, teacher, labels, temperature=1.0)
loss.backward()
print("KL", round(loss.item(), 4))
print("學生分數梯度", student.grad.round(decimals=4).tolist())
print("教師梯度", teacher.grad)
