import torch

should_refuse = torch.tensor([True, False, True, False])
actually_refuse = torch.ones(4, dtype=torch.bool)
refusal_rate = actually_refuse[should_refuse].float().mean()
completion_rate = (~actually_refuse[~should_refuse]).float().mean()
print("該拒絕的拒絕比例", refusal_rate.item())
print("正常題未拒絕比例", completion_rate.item())
