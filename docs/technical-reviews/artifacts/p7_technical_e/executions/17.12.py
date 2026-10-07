import torch
calibration = torch.tensor([-0.5, 0.2, 0.8, 1.0])
evaluation = torch.tensor([-0.2, 0.4, 3.0])
fixed_max = calibration.abs().max()
dynamic_max = evaluation.abs().max()
def simulate(x, maximum):
    scale = maximum / 7
    return (x / scale).round().clamp(-7, 7) * scale
for name, maximum in (("固定", fixed_max), ("動態", dynamic_max)):
    restored = simulate(evaluation, maximum)
    overflow = (evaluation.abs() > maximum).sum().item()
    print(name, "範圍", maximum.item(), "超出項數", overflow, "還原", restored.round(decimals=4).tolist())
