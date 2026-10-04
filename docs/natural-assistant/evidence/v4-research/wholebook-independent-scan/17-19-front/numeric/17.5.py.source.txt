import torch
from tiny_perceptron.quantization import quantize_symmetric

w = torch.tensor([[0.01, 0.03, 0.05, 0.08], [10.0, 30.0, 50.0, 80.0]])
for per_row in (False, True):
    q, scale = quantize_symmetric(w, bits=4, per_channel=per_row)
    restored = q.float() * scale
    error = (w - restored).abs().mean(-1)
    print("每列獨立", per_row, "整數", q.tolist())
    print("每列MAE", error.round(decimals=4).tolist(), "scale格數", scale.numel())
