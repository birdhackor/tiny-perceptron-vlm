import torch

valid = torch.tensor([[True, True, False, False], [False, True, True, True]])
positions = torch.arange(4)[None].expand_as(valid)
last = positions.masked_fill(~valid, -1).amax(dim=-1)
logits = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
selected = logits[torch.arange(2), last]
print("最後有效索引", last)
print("取出的候選分數", selected)
assert last.tolist() == [1, 3]
