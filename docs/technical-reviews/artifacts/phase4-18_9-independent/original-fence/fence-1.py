import torch
from tiny_perceptron.alignment import distillation_loss, distillation_kl
from tiny_perceptron.model import masked_loss

student = torch.zeros(1, 2, 2, requires_grad=True)
p = [0.8, 0.2]
teacher = torch.tensor([[p, p]]).log()
labels = torch.tensor([[0, 1]])
ce = masked_loss(student, labels)
k = distillation_kl(student, teacher, labels, temperature=1.0)
print("CE/KL", round(ce.item(), 4), round(k.item(), 4))
for alpha in (0.0, 0.5, 1.0):
    loss = distillation_loss(student, teacher, labels, alpha=alpha, temperature=1.0)
    print("alpha", alpha, "總誤差", round(loss.item(), 4))
assert torch.allclose(distillation_loss(student, teacher, labels, alpha=0, temperature=1), ce)
