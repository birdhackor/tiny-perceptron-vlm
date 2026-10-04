import torch

power = torch.tensor([0.0, 1e-10, 1e-4, 1.0])
floor = 1e-8
compressed = power.clamp(min=floor).log()
scaled = (4 * power).clamp(min=floor).log()
print("log功率", [round(x, 2) for x in compressed.tolist()])
print("放大後差", [round(x, 2) for x in (scaled - compressed).tolist()])
print("皆為有限值", bool(compressed.isfinite().all()))
