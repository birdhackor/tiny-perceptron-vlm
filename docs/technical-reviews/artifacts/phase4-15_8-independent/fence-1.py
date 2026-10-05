import torch

for advantage in (8.0, 0.1):
    scores = torch.tensor([[advantage, 0.0, 0.0]]).repeat(12, 1)
    prob = scores.softmax(-1)
    chosen = prob.topk(1, dim=-1).indices
    counts = torch.bincount(chosen.flatten(), minlength=3)
    print("分數優勢", advantage)
    print("平均比例", prob.mean(0).round(decimals=4).tolist())
    print("工作次數", counts.tolist())
