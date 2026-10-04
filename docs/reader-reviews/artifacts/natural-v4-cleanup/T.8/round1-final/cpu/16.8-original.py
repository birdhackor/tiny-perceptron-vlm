import torch
import torch.nn.functional as F

torch.manual_seed(0)
q, k, v = [torch.randn(1, 1, 4, 3) for _ in range(3)]
allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
scores = q @ k.transpose(-2, -1) / (3**0.5)
weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
manual = weights @ v
optimized = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
print("輸出形狀", tuple(optimized.shape))
print("最大差異", (manual - optimized).abs().max().item())
assert torch.allclose(manual, optimized, atol=1e-6)
