import torch

x = torch.tensor([-0.7, 0.2, 0.7, 1.2])
scale = 0.25
q = (x / scale).round().clamp(-127, 127).to(torch.int8)
restored = q.float() * scale
print("整數格", q.tolist())
print("還原值", restored.tolist())
print("差值", (x - restored).round(decimals=4).tolist())
