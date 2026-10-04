import torch
from torch import nn

torch.manual_seed(0)
vision = torch.randn(1, 16, 8)
projector = nn.Linear(8, 12)
language_features = projector(vision)
print("視覺", tuple(vision.shape))
print("文字介面", tuple(language_features.shape))
print("接頭矩陣", tuple(projector.weight.shape))
