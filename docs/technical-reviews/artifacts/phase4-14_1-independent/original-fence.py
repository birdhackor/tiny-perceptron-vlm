import math
import torch


def rotate(x, position):
    angle = position * math.pi / 4
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([c * x[0] - s * x[1], s * x[0] + c * x[1]])


q = torch.tensor([1.0, 0.0])
k = torch.tensor([1.0, 0.0])


def score(a, b):
    return torch.dot(rotate(q, a), rotate(k, b)).item()


print(round(score(2, 5), 4))
print(round(score(12, 15), 4))
print(round(score(3, 5), 4))
