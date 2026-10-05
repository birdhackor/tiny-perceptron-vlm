"""Bounded G.1 checks; fixed inputs, no training/data/model downloads."""
from pathlib import Path
import hashlib
import json
import platform
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()

vocab = ["狗", "看", "貓"]
to_id = {text: index for index, text in enumerate(vocab)}
ids = [to_id[text] for text in "貓看狗"]
decoded = "".join(vocab[index] for index in ids)
assert ids == [2, 1, 0] and decoded == "貓看狗"
emb = nn.Embedding(3, 2, dtype=torch.float64)
with torch.no_grad():
    emb.weight.copy_(torch.tensor([[0.0, 0.5], [0.8, 0.3], [0.2, -0.1]], dtype=torch.float64))
looked_up = emb(torch.tensor(ids))
assert torch.equal(looked_up[0], torch.tensor([0.2, -0.1], dtype=torch.float64))
assert emb.weight.requires_grad and tuple(looked_up.shape) == (3, 2)

# Change only the ID convention and corresponding table order.
new_vocab = ["貓", "狗", "看"]
new_ids = [new_vocab.index(text) for text in "貓看狗"]
new_emb = nn.Embedding(3, 2, dtype=torch.float64)
with torch.no_grad():
    new_emb.weight.copy_(emb.weight[torch.tensor([2, 0, 1])])
assert new_ids == [0, 2, 1]
assert torch.equal(new_emb(torch.tensor(new_ids)), looked_up)
assert "".join(new_vocab[index] for index in new_ids) == decoded

# A gradient is computed once; no optimizer or parameter update is run.
emb(torch.tensor([2])).sum().backward()
assert torch.equal(emb.weight.grad, torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float64))
scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
assert tuple(scores.shape) == (2, 3) and scores.numel() == 6
assert tuple(scores.T.shape) == (3, 2)

linear = nn.Linear(3, 2, dtype=torch.float64)
with torch.no_grad():
    linear.weight.copy_(torch.tensor([[1.0, 0.0, 2.0], [0.0, -1.0, 1.0]], dtype=torch.float64))
    linear.bias.copy_(torch.tensor([1.0, 0.0], dtype=torch.float64))
x = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
out = linear(x)
assert torch.equal(out, torch.tensor([[10.0, 2.0]], dtype=torch.float64))
assert torch.equal(out, x @ linear.weight.T + linear.bias)
probe = torch.tensor([-2.0, 0.0, 2.0, 3.0], dtype=torch.float64)
hidden = torch.stack([probe, -probe], dim=1)
pure = hidden.sum(dim=1)
activated = nn.ReLU()(hidden).sum(dim=1)
assert torch.equal(pure, torch.zeros_like(probe))
assert torch.equal(activated, torch.tensor([2.0, 0.0, 2.0, 3.0], dtype=torch.float64))
# An affine function through (-2, 2) and (0, 0) has slope -1 and predicts -2 at 2.
k = (0.0 - 2.0) / (0.0 - (-2.0))
assert k * 2.0 == -2.0 and k * 2.0 != 2.0

# Execute the exact inspected official bpe_encode body, with visualization disabled.
bpe_source = HERE / "sources/tiktoken-educational-bpe_encode.txt"
namespace = {"__name__": "official_bpe_excerpt"}
exec(compile(bpe_source.read_bytes(), str(bpe_source), "exec"), namespace)
bpe_encode = namespace["bpe_encode"]
raw = "貓看".encode("utf-8")
base = {bytes([b]): b for b in range(256)}
byte_tokens = bpe_encode(base, raw, visualise=None)
assert byte_tokens == [232, 178, 147, 231, 156, 139]
partial = {**base, raw[:2]: 256}
partial_tokens = bpe_encode(partial, raw, visualise=None)
assert partial_tokens == [256, 147, 231, 156, 139]
merged = dict(base)
for length in range(2, len(raw) + 1):
    merged[raw[:length]] = 254 + length
merged_tokens = bpe_encode(merged, raw, visualise=None)
assert merged_tokens == [260]
for ranks, tokens in [(base, byte_tokens), (partial, partial_tokens), (merged, merged_tokens)]:
    inverse = {token: token_bytes for token_bytes, token in ranks.items()}
    assert b"".join(inverse[token] for token in tokens).decode("utf-8") == "貓看"
try:
    raw[:2].decode("utf-8")
except UnicodeDecodeError:
    incomplete_bytes = True
else:
    incomplete_bytes = False
assert incomplete_bytes

result = {
    "id_mapping": {"vocab": vocab, "text": decoded, "ids": ids, "permuted_ids": new_ids, "permuted_embedding_equal": True},
    "embedding": {"ID_2_vector": looked_up[0].tolist(), "requires_grad": emb.weight.requires_grad, "lookup_shape": list(looked_up.shape), "single_ID_2_sum_gradient": emb.weight.grad.tolist(), "optimizer_steps": 0},
    "shape": {"two_rows_three_columns": list(scores.shape), "elements": scores.numel(), "transposed": list(scores.T.shape)},
    "linear_activation": {"linear_output": out.tolist(), "input_values": probe.tolist(), "pure_linear": pure.tolist(), "relu_then_sum": activated.tolist(), "affine_prediction_at_2_from_first_two_points": k * 2.0},
    "tokenizer": {"text": "貓看", "code_points": 2, "UTF8_bytes": list(raw), "byte_token_ids": byte_tokens, "partial_byte_BPE_ids": partial_tokens, "multi_character_BPE_ids": merged_tokens, "roundtrips": 3, "vocabularies": "three manually fixed mappings; no fitting or tokenizer training", "executed_original_function_sha256": hashlib.sha256(bpe_source.read_bytes()).hexdigest()},
    "environment": {"python": sys.version, "executable": sys.executable, "pytorch": torch.__version__, "pytorch_git": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads()), "platform": platform.platform()},
    "scope": "G.1 mapping, lookup, shape and nonlinear representation only; deterministic fixed-vocabulary BPE mechanics, no model ability/quality evaluation, no training or saved weights.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
