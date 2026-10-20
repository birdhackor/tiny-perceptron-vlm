import torch
from torch import nn
from tiny_perceptron.data import IGNORE
from tiny_perceptron.multimodal import expand_modalities

embedding = nn.Embedding(264, 8)
ids = torch.tensor([1, 5, 4, 73, 2])
labels = torch.tensor([IGNORE, IGNORE, IGNORE, 73, 2])
features = {5: torch.randn(3, 8)}
x, y = expand_modalities(ids, labels, embedding, features, {5})
print("輸入特徵", tuple(x.shape))
print("下一位置目標", y.tolist())
