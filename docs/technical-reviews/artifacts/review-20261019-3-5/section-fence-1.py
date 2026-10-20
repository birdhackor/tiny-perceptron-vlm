import math
import torch

torch.manual_seed(42)
for d in [4, 64]:
    q = torch.randint(0, 2, (1000, d)).float() * 2 - 1
    k = torch.randint(0, 2, (1000, d)).float() * 2 - 1
    scores = (q * k).sum(dim=-1)
    scaled = scores / math.sqrt(d)
    print(d, scores.std().item(), scaled.std().item())
