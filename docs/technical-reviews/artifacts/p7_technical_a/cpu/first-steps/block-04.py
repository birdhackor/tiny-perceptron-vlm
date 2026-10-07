from torch import nn

layer = nn.Linear(3, 2)
with torch.no_grad():
    layer.weight.copy_(weights.T)
    layer.bias.copy_(torch.tensor([5.0, 0.0]))
print(layer(recipes).tolist())
