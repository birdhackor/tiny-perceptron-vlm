import torch
from tiny_perceptron.alignment import sequence_log_probability
logits = torch.tensor([[[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])
labels = torch.tensor([[-100, 0, 2]])
score = sequence_log_probability(logits, labels)
print("整段log分數", round(score.item(), 4))
print("還原機率", round(score.exp().item(), 4))
print('torch',torch.__version__)
