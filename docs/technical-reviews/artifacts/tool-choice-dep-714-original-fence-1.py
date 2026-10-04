import torch
from torch.nn import functional as F

logits = torch.tensor([[0.0, 0.0, 3.0, 1.0]])
for label in [2, 3]:
    loss = F.cross_entropy(logits, torch.tensor([label]))
    print("提供的答案ID", label, "代價", loss.item())
wrong_favored = torch.tensor([[0.0, 0.0, 3.0, 5.0]])
print("提高錯字分數後，錯標代價", F.cross_entropy(wrong_favored, torch.tensor([3])).item())
