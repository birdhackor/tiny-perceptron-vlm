import torch
from tiny_perceptron.model import masked_loss

a = torch.tensor([[[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
labels = torch.tensor([[1, 2]])
b = torch.cat([a, torch.zeros(1, 3, 4)], dim=1)
extended = torch.tensor([[1, 2, -100, -100, -100]])
before = masked_loss(a, labels)
after = masked_loss(b, extended)
print(before.item(), after.item())
assert torch.allclose(before, after)
