import torch

from tiny_perceptron.capstone import CapstoneModel

torch.manual_seed(42)
core = CapstoneModel().language
core.eval()
ids = torch.tensor([[1, 21, 22, 23, 24]])
with torch.no_grad():
    full = core(ids)["logits"][:, -1]
    cache = core(ids[:, :3])["cache"]
    cached = core(ids[:, 3:], cache=cache)["logits"][:, -1]
print("最大分數差", (full - cached).abs().max().item(), "接近", torch.allclose(full, cached, atol=1e-4, rtol=1e-4))
