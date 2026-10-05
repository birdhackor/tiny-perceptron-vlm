import torch
from torch import nn

scores = nn.Embedding(3, 3)
with torch.no_grad():
    scores.weight.zero_()
    scores.weight[0, 1] = 2.0
    scores.weight[2, 0] = 3.0
inputs = torch.tensor([0, 2])
output = scores(inputs)
print(scores.weight)
print(output)
