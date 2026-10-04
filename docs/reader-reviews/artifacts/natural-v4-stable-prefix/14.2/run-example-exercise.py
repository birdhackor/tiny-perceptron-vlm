import os
os.environ["OMP_NUM_THREADS"]="1"
import torch
import torch.nn.functional as F

x = torch.tensor([[2.0, 2.0, 2.0], [1.0, 2.0, 3.0]])
eps = 1e-5
ln = F.layer_norm(x, (3,), eps=eps)
rms = x / torch.sqrt(x.square().mean(-1, keepdim=True) + eps)
print("LN", ln.round(decimals=4))
print("RMS", rms.round(decimals=4))

print("INPUT_SHAPE", tuple(x.shape))
print("LN_SHAPE", tuple(ln.shape))
print("RMS_SHAPE", tuple(rms.shape))
shifted = x + 10
shifted_ln = F.layer_norm(shifted, (3,), eps=eps)
shifted_rms = shifted / torch.sqrt(shifted.square().mean(-1, keepdim=True) + eps)
print("SHIFTED_INPUT", shifted)
print("SHIFTED_LN", shifted_ln.round(decimals=4))
print("SHIFTED_RMS", shifted_rms.round(decimals=4))
print("LN_UNCHANGED", torch.allclose(ln, shifted_ln))
print("RMS_UNCHANGED", torch.allclose(rms, shifted_rms))
for row in range(2):
    print("ROW", row, "LN_UNCHANGED", torch.allclose(ln[row], shifted_ln[row]), "RMS_UNCHANGED", torch.allclose(rms[row], shifted_rms[row]))
print("LN_MAX_ABS_DIFF", (ln-shifted_ln).abs().max().item())
print("RMS_MAX_ABS_DIFF", (rms-shifted_rms).abs().max().item())
