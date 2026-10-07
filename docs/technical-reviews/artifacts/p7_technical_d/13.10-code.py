import torch
from tiny_perceptron.posttraining import preference_loss
for preferred_score in [0.0, 2.0]:
    chosen = torch.tensor([preferred_score]);rejected = torch.tensor([0.0]);gap = chosen - rejected
    win_probability = torch.sigmoid(gap);loss = preference_loss(chosen, rejected)
    print("差距", gap.item(), "勝出機率", round(win_probability.item(), 4), "代價", round(loss.item(), 4))
print('torch',torch.__version__)
