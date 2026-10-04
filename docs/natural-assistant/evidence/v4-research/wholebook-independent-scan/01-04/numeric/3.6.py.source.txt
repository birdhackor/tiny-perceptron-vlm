import torch
from tiny_perceptron.attention import manual_attention

q = torch.zeros(1, 1, 4, 2)
k = torch.zeros_like(q)
v = torch.tensor([[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]]])
allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
out, weights = manual_attention(q, k, v, allowed)
changed = v.clone()
changed[:, :, 3] += 100
other, _ = manual_attention(q, k, changed, allowed)
print(weights[0, 0])
print(out[0, 0])
assert torch.allclose(out[:, :, :3], other[:, :, :3])
