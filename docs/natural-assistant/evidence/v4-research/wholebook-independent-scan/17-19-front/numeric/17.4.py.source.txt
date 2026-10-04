import torch

x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
low = min(x.min().item(), 0.0)
high = max(x.max().item(), 0.0)
scale = (high - low) / 15
zero = round(-low / scale)
q = (x / scale + zero).round().clamp(0, 15).to(torch.uint8)
restored = (q.float() - zero) * scale
print("scale/zero", scale, zero)
print("整數碼", q.tolist())
print("還原", restored.round(decimals=4).tolist())
