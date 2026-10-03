import torch
from tiny_perceptron.quantization import quantize_symmetric

x = torch.tensor([0.1, 0.2, 0.3, 100.0])
for clip in (False, True):
    source = x.clamp(-1, 1) if clip else x
    q, scale = quantize_symmetric(source, bits=4)
    restored = q.float() * scale
    error = (x - restored).abs()
    print("截斷", clip, "還原", restored.round(decimals=4).tolist())
    print(
        "小值MAE",
        round(error[:3].mean().item(), 4),
        "極值差",
        round(error[-1].item(), 4),
        "全部MAE",
        round(error.mean().item(), 4),
    )
