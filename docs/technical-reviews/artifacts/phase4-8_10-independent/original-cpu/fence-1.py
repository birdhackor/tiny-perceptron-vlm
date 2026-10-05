import torch

human = torch.tensor([1, 0, 1, 0])
recorded_judge = torch.tensor([1, 1, 1, 0])
agree = human == recorded_judge
print("一致比例", agree.float().mean().item())
print("分歧索引", (~agree).nonzero().flatten().tolist())
