import torch
from tiny_perceptron.quantization import quantize_symmetric

w = torch.arange(-7, 8).float() / 7
for bits in (8, 4):
    q, scale = quantize_symmetric(w, bits)
    restored = q.float() * scale
    error = (w - restored).abs()
    packed_bytes = (w.numel() * bits + 7) // 8
    print("bits", bits, "MAE", round(error.mean().item(), 6), "最大差", round(error.max().item(), 6))
    print("理想整數bytes", packed_bytes, "目前q容器bytes", q.numel() * q.element_size())
