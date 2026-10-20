"""核對多頭注意力算例、拆頭順序與精簡零件的實際行為。"""

import contextlib
import hashlib
import inspect
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.attention import CausalAttention as LibraryAttention
from tiny_perceptron.attention import manual_attention

source = ROOT / sys.argv[1]
raw = source.read_bytes()
blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(blocks[0], str(source), "exec"), namespace)
    exec(compile(blocks[1], str(source), "exec"), namespace)
Attention = namespace["CausalAttention"]
torch.set_num_threads(1)
torch.set_default_device("cpu")
results = {}

features = namespace["features"]
split = namespace["split"]
assert features.tolist() == [list(map(list, (range(8), range(8, 16))))]
assert split.shape == (1, 2, 2, 4)
assert split[0, 0].tolist() == [[0, 1, 2, 3], [8, 9, 10, 11]]
assert split[0, 1].tolist() == [[4, 5, 6, 7], [12, 13, 14, 15]]
results["拆頭"] = {"shape": list(split.shape), "values": split.tolist()}

values = torch.tensor([[[10., 0, 0, 0], [0, 20., 0, 0]],
                       [[4., 8., 0, 0], [12., 0, 0, 0]]])
weights = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
mixed = torch.einsum("ht,htd->hd", weights, values)
assert mixed.tolist() == [[7.5, 5., 0., 0.], [10., 2., 0., 0.]]
joined = mixed.reshape(8)
projection = nn.Linear(8, 8, bias=False)
with torch.no_grad():
    projection.weight.copy_(torch.eye(8))
    projection.weight[0, 4] = 1
projected = projection(joined)
assert projected[0].item() == 17.5
results["指定比例與跨頭混合"] = {
    "values": values.tolist(), "weights": weights.tolist(),
    "mixed": mixed.tolist(), "joined": joined.tolist(),
    "projection_first_component": projected[0].item(),
    "derivation": "頭零為 3/4×10=7.5、1/4×20=5；頭一為 1/4×4+3/4×12=10、1/4×8=2。投影示例第一格為 7.5+10=17.5。",
}

comparison = []
for heads in (1, 2, 4):
    torch.manual_seed(42)
    layer = Attention(width=8, heads=heads)
    x = torch.randn(1, 4, 8)
    out = layer(x)
    library = LibraryAttention(width=8, heads=heads)
    loaded = library.load_state_dict(layer.state_dict())
    library_out, cache = library(x)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert list(layer.state_dict()) == ["q.weight", "k.weight", "v.weight", "out.weight"]
    assert len(list(layer.parameters())) == 4
    assert all(m.bias is None for m in (layer.q, layer.k, layer.v, layer.out))
    assert out.shape == x.shape == (1, 4, 8)
    assert torch.allclose(out, library_out)
    assert all(torch.equal(layer.state_dict()[k], library.state_dict()[k]) for k in layer.state_dict())
    assert all(layer.state_dict()[k].data_ptr() != library.state_dict()[k].data_ptr() for k in layer.state_dict())
    assert isinstance(cache, tuple) and len(cache) == 2
    assert tuple(cache[0].shape) == (1, heads, 4, 8 // heads)
    changed = x.clone()
    changed[:, 3, :] += 100
    assert torch.equal(layer(changed)[:, :3], out[:, :3])
    comparison.append({"heads": heads, "head_dim": layer.head_dim,
                       "input_shape": list(x.shape), "output_shape": list(out.shape),
                       "keys": list(layer.state_dict()), "same": bool(torch.allclose(out, library_out)),
                       "max_absolute_difference": float((out-library_out).abs().max().detach()),
                       "cache_shapes": [list(z.shape) for z in cache]})
results["零件與函式庫比較"] = comparison

try:
    Attention(width=8, heads=3)
except ValueError as error:
    results["不能均分"] = {"exception": type(error).__name__, "message": str(error), "remainder": 8 % 3}
else:
    raise AssertionError("不能均分時應拒絕建立注意力零件")
default = Attention(width=8)
assert default.heads == 1 and default.head_dim == 8
assert issubclass(nn.Linear, nn.Module) and issubclass(nn.Embedding, nn.Module)

class MissingParentInit(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(3, 2)

try:
    MissingParentInit()
except AttributeError as error:
    results["父類別初始化"] = {"exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("未初始化父類別時不應能登記 Linear")

class ForwardProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + 1

probe = ForwardProbe()
assert probe(torch.tensor(2)).item() == 3 and probe.calls == 1
results["類別與參數"] = {
    "default_heads": default.heads, "default_head_dim": default.head_dim,
    "embedding_and_linear_are_modules": True, "module_call_forward_calls": probe.calls,
    "library_init_signature": str(inspect.signature(LibraryAttention)),
    "library_forward_signature": str(inspect.signature(LibraryAttention.forward)),
}

torch.manual_seed(42)
q, k, v = [torch.randn(2, 2, 4, 4, dtype=torch.float64) for _ in range(3)]
allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
out, attention_weights = manual_attention(q, k, v, allowed)
per_head = torch.empty_like(out)
for batch in range(2):
    for head in range(2):
        scores = q[batch, head] @ k[batch, head].T / math.sqrt(4)
        scores = scores.masked_fill(~allowed[0, 0], -math.inf)
        per_head[batch, head] = scores.softmax(-1) @ v[batch, head]
assert torch.allclose(out, per_head, atol=1e-12, rtol=1e-12)
assert torch.count_nonzero(attention_weights.triu(1)).item() == 0
restored = out.transpose(1, 2).reshape(2, 4, 8)
contiguous_view = out.transpose(1, 2).contiguous().view(2, 4, 8)
assert torch.equal(restored, contiguous_view)
assert torch.equal(restored, torch.cat((out[:, 0], out[:, 1]), dim=-1))
matrix = torch.arange(6).reshape(2, 3)
assert torch.equal(matrix.T, matrix.transpose(0, 1))
torch.manual_seed(42)
random_first = torch.randn(1, 4, 8)
torch.manual_seed(42)
random_second = torch.randn(1, 4, 8)
assert torch.equal(random_first, random_second)
results["分頭計算與合併"] = {
    "allowed": allowed.tolist(), "scale": math.sqrt(4),
    "per_head_max_absolute_difference": float((out-per_head).abs().max()),
    "future_weights_nonzero": int(torch.count_nonzero(attention_weights.triu(1))),
    "reshape_equals_contiguous_view": bool(torch.equal(restored, contiguous_view)),
    "head_order_equals_cat": bool(torch.equal(restored, torch.cat((out[:, 0], out[:, 1]), -1))),
    "matrix_T_equals_transpose": True, "same_environment_seed_replay": True,
}

torch.manual_seed(42)
sample = torch.randn(1, 4, 8)
reference = LibraryAttention(8, heads=2)
reference_out, _ = reference(sample)
gqa = LibraryAttention(8, heads=2, kv_heads=1)
_, compact_cache = gqa(sample)
assert compact_cache[0].shape == (1, 1, 4, 4)
sdpa = LibraryAttention(8, heads=2, backend="sdpa")
sdpa.load_state_dict(reference.state_dict())
sdpa_out, _ = sdpa(sample)
assert torch.allclose(reference_out, sdpa_out, atol=1e-6, rtol=1e-5)
prefix, cache = reference(sample[:, :2])
suffix, _ = reference(sample[:, 2:], cache=cache)
cached_out = torch.cat((prefix, suffix), dim=1)
assert torch.allclose(reference_out, cached_out, atol=1e-6, rtol=1e-5)
rotary = LibraryAttention(8, heads=2, rotary=True)
rotary.load_state_dict(reference.state_dict())
positioned, _ = rotary(sample, positions=torch.tensor([0, 1, 4, 9]))
ordinary, _ = rotary(sample)
assert not torch.allclose(positioned, ordinary)
from tiny_perceptron.attention import attention_mask

positions = torch.arange(4)
valid = torch.tensor([[False, True, True, True]])
segments = torch.tensor([[0, 0, 1, 1]])
pad_mask = attention_mask(positions, positions, valid=valid)
document_mask = attention_mask(positions, positions, segments=segments)
assert not pad_mask[..., 0].any()
assert not document_mask[0, 0, 2:, :2].any()
assert document_mask[0, 0, 3, 2]
results["函式庫選項"] = {
    "compact_cache_shape": list(compact_cache[0].shape),
    "sdpa_max_absolute_difference": float((reference_out-sdpa_out).abs().max().detach()),
    "cached_max_absolute_difference": float((reference_out-cached_out).abs().max().detach()),
    "custom_rotary_positions_change_output": True,
    "padding_keys_excluded": True, "cross_document_keys_excluded": True,
}

print(json.dumps({"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
                  "docs/technical-reviews/artifacts/review-20261019-3-7/verify.py " + sys.argv[1],
                  "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                  "torch_git_version": str(torch.version.git_version),
                                  "device": "cpu", "numeric_dtype": "float64", "module_dtype": "float32"},
                  "source_sha256": hashlib.sha256(raw).hexdigest(),
                  "captured_split_stdout": captured.getvalue(), "results": results,
                  "result": "全部斷言通過"}, ensure_ascii=False, indent=2))
