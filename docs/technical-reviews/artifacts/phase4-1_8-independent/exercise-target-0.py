import torch
from torch.nn import functional as F

p = torch.tensor([0.1, 0.5, 0.9])
print("正確機率", p, "代價", -p.log())
logits = torch.tensor([[0.0, 2.0, 0.0]])
target = torch.tensor([0])
probabilities = logits.softmax(dim=-1)
manual = -probabilities[0, 0].log()
loss = F.cross_entropy(logits, target)
print(probabilities, manual.item(), loss.item())
assert torch.allclose(loss, manual)
