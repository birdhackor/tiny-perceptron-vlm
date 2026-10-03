import math
import torch

x = torch.tensor([[1.0, 2.0]])
router_weight = torch.tensor([[0.0, 0.0, math.log(2)], [0.0, 0.0, 0.0]])
scores = x @ router_weight
prob = scores.softmax(-1)
expert_outputs = torch.stack([x, 2 * x, 3 * x], dim=1)
output = (prob[:, :, None] * expert_outputs).sum(1)
print("分配比例", prob.tolist())
print("混合輸出", output.tolist())
