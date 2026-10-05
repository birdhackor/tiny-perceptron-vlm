import torch
from tiny_perceptron.alignment import dpo_loss

reference = torch.tensor([0.5, 0.5]).log()
for probabilities in [[0.5, 0.5], [0.7, 0.3]]:
    policy = torch.tensor(probabilities).log()
    loss = dpo_loss(policy[:1], policy[1:], reference[:1], reference[1:], beta=1.0)
    print("較佳／較差機率", probabilities, "偏好代價", round(loss.item(), 4))
