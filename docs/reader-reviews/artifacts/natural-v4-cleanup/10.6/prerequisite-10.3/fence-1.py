import torch
from torch import nn
from tiny_perceptron.multimodal import scene, patchify

torch.manual_seed(0)
patches = patchify(scene()[None], 4)
embedding = nn.Linear(48, 8)
features = embedding(patches)
print("輸入", tuple(patches.shape))
print("輸出", tuple(features.shape))
print("矩陣", tuple(embedding.weight.shape))
