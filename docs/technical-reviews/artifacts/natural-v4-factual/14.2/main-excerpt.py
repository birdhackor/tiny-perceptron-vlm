import torch
import torch.nn.functional as F

x = torch.tensor([[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]])
eps = 1e-5
ln = F.layer_norm(x, (3,), eps=eps)
rms = x / torch.sqrt(x.square().mean(-1, keepdim=True) + eps)
print("LN", ln.round(decimals=4))
print("RMS", rms.round(decimals=4))
