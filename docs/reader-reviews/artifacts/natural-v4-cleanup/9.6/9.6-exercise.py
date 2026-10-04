import torch

should_refuse = torch.tensor([True, False, True, False])
cases = [
    ("exercise aligned", [True, False, True, False]),
    ("exercise first observation changed", [False, False, True, False]),
    ("prose all nonrefusal", [False, False, False, False]),
]
for name, observations in cases:
    actually_refuse = torch.tensor(observations)
    refusal_rate = actually_refuse[should_refuse].float().mean()
    completion_rate = (~actually_refuse[~should_refuse]).float().mean()
    print(name, observations)
    print("該拒絕的拒絕比例", refusal_rate.item())
    print("正常題未拒絕比例", completion_rate.item())
