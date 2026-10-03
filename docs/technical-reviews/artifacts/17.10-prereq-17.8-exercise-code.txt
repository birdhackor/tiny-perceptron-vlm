import torch
from tiny_perceptron.quantization import pack_int4, unpack_int4

q = torch.tensor([-8, -1, 0, 1, 7, 2, 3, 4], dtype=torch.int8)
packed = pack_int4(q)
restored = unpack_int4(packed, q.shape)
print("原數字", q.tolist())
print("packed", packed.tolist(), "bytes", packed.numel() * packed.element_size())
print("拆回", restored.tolist())
assert torch.equal(q, restored)
