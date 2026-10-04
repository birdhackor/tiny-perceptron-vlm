import torch

x = torch.tensor([[0.7, 0.2], [-0.7, 1.2]])
w = torch.tensor([[0.7, 0.2], [-0.7, 1.2]])
scale = 0.5


def simulate_quantization(t):
    return (t / scale).round() * scale


qx, qw = simulate_quantization(x), simulate_quantization(w)
reference = x @ w.T
weight_only = x @ qw.T
both = qx @ qw.T
print("原輸出", reference.round(decimals=4).tolist())
for name, output in (("只改權重", weight_only), ("權重加特徵", both)):
    print(name, output.round(decimals=4).tolist(), "MAE", round((reference - output).abs().mean().item(), 4))
