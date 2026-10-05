import torch
from torch import nn

torch.manual_seed(0)
projector = nn.Linear(4, 8)
a, b = torch.zeros(1, 4), torch.ones(1, 4)
with torch.no_grad():
    projector.weight.zero_()
difference = (projector(a) - projector(b)).norm()
print("輸出形狀", tuple(projector(a).shape))
print("對輸入差異敏感", difference.item() > 0)
