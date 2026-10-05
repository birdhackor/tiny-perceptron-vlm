import torch
from tiny_perceptron.alignment import dpo_loss

for beta in [0.1, 1.0, 5.0]:
    chosen = torch.tensor([-2.0], requires_grad=True)
    loss = dpo_loss(chosen, torch.tensor([-4.0]), torch.tensor([-3.0]), torch.tensor([-3.0]), beta=beta)
    loss.backward()
    print(beta, "代價", round(loss.item(), 6), "梯度", round(chosen.grad.item(), 6))
