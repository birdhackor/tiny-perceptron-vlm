"""核對因果注意力算例、空列與函式庫呼叫。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron import attention

torch.set_num_threads(1)
torch.set_default_device("cpu")
source = ROOT / "docs/course-review-20261019/freeze-01/sources/3.6.md"
raw = source.read_bytes()
namespace = {"__name__": "__main__"}
captured = io.StringIO()
fences = re.findall(r"```python\n(.*?)\n```", raw.decode(), re.S)
with contextlib.redirect_stdout(captured):
    for fence in fences:
        exec(compile(fence, str(source), "exec"), namespace)


def record_tensor(value):
    return value.detach().cpu().tolist()


def close(actual, expected):
    assert torch.allclose(actual, torch.tensor(expected, dtype=actual.dtype), atol=1e-6, rtol=0.0)


manual = namespace["manual_attention"]
q, k, v, allowed = [namespace[name] for name in ("q", "k", "v", "allowed")]
out, weights = manual(q, k, v, allowed)
assert list(q.shape) == list(k.shape) == list(v.shape) == [1, 1, 4, 2]
assert list(allowed.shape) == [1, 1, 4, 4]
assert allowed.dtype == torch.bool
assert list((q @ k.transpose(-2, -1)).shape) == [1, 1, 4, 4]
assert bool(((q @ k.transpose(-2, -1)) == 0).all())
close(weights[0, 0], [[1, 0, 0, 0], [.5, .5, 0, 0], [1/3, 1/3, 1/3, 0], [.25]*4])
close(out[0, 0], [[1, 10], [1.5, 15], [2, 20], [2.5, 25]])

changed = v.clone()
changed[:, :, 3] += 100
other, _ = manual(q, k, changed, allowed)
close(v[0, 0], [[1, 10], [2, 20], [3, 30], [4, 40]])
close(changed[0, 0, 3], [104, 140])
close(other[0, 0], [[1, 10], [1.5, 15], [2, 20], [27.5, 50]])
assert torch.equal(out[:, :, :3], other[:, :, :3])

all_allowed = torch.ones(4, 4, dtype=torch.bool)[None, None]
all_out, all_weights = manual(q, k, v, all_allowed)
all_other, _ = manual(q, k, changed, all_allowed)
close(all_weights[0, 0], [[.25]*4]*4)
close(all_out[0, 0], [[2.5, 25]]*4)
close(all_other[0, 0], [[27.5, 50]]*4)
assertion_failed = False
try:
    assert torch.allclose(all_out[:, :, :3], all_other[:, :, :3])
except AssertionError:
    assertion_failed = True
assert assertion_failed

empty_allowed = allowed.clone()
empty_allowed[0, 0, 0, 0] = False
empty_out, empty_weights = manual(q, k, v, empty_allowed)
assert torch.equal(empty_weights[:, :, 0], torch.zeros_like(empty_weights[:, :, 0]))
assert torch.equal(empty_out[:, :, 0], torch.zeros_like(empty_out[:, :, 0]))
assert torch.equal(empty_out[:, :, 1:], out[:, :, 1:])
assert torch.isfinite(empty_out).all() and torch.isfinite(empty_weights).all()
raw_empty_softmax = torch.full((4,), float("-inf")).softmax(-1)
assert torch.isnan(raw_empty_softmax).all()
assert math.exp(float("-inf")) == 0.0
assert torch.exp(torch.tensor(float("-inf"))).item() == 0.0
zero_fill_weights = (q @ k.transpose(-2, -1)).masked_fill(~allowed, 0.0).softmax(-1)
close(zero_fill_weights[0, 0], [[.25]*4]*4)
assert list(allowed.any(dim=-1, keepdim=True).shape) == [1, 1, 4, 1]
assert torch.equal(~torch.tensor([True, False]), torch.tensor([False, True]))

repo_tree = ast.parse((ROOT / "tiny_perceptron/attention.py").read_text())
repo_manual = next(n for n in repo_tree.body if isinstance(n, ast.FunctionDef) and n.name == "manual_attention")
excerpt_tree = ast.parse(fences[1])
assert ast.dump(repo_manual) == ast.dump(excerpt_tree.body[0])
repo_out, repo_weights = attention.manual_attention(q, k, v, allowed)
assert torch.equal(repo_out, out) and torch.equal(repo_weights, weights)

position = torch.arange(5)
valid = torch.tensor([[False, False, True, True, True]])
pad_allowed = attention.attention_mask(position, position, valid=valid)
pad_out, pad_weights = attention.manual_attention(torch.zeros(1, 1, 5, 2), torch.zeros(1, 1, 5, 2), torch.ones(1, 1, 5, 2), pad_allowed)
assert not pad_allowed[:, :, :2].any()
assert not pad_allowed[:, :, :, :2].any()
assert not pad_out[:, :, :2].any()
assert torch.isfinite(pad_out).all()

segment = torch.tensor([[0, 0, 1, 1]])
segment_allowed = attention.attention_mask(torch.arange(4), torch.arange(4), segments=segment)
assert segment_allowed[0, 0].int().tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
combined = attention.attention_mask(torch.arange(4), torch.arange(4), valid=torch.tensor([[True, True, True, False]]), segments=segment)
assert combined[0, 0].int().tolist() == [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]

torch.manual_seed(42)
layer = attention.CausalAttention(width=8, heads=1)
before = {key: value.clone() for key, value in layer.state_dict().items()}
with patch.object(attention, "manual_attention", wraps=attention.manual_attention) as manual_spy:
    with patch.object(attention, "attention_mask", wraps=attention.attention_mask) as mask_spy:
        library_output, _ = layer(torch.randn(1, 4, 8), valid=torch.tensor([[True, True, True, False]]), segments=segment)
assert manual_spy.call_count == mask_spy.call_count == 1
assert torch.isfinite(library_output).all()
assert all(torch.equal(before[key], value) for key, value in layer.state_dict().items())

chapter = (ROOT / "course/chapters/03.md").read_text()
next_section = chapter.split("## 3.7 ", 1)[1].split("\n## ", 1)[0]
class_fence = next(fence for fence in re.findall(r"```python\n(.*?)\n```", next_section, re.S) if "class CausalAttention" in fence)
class_namespace = {"torch": torch}
exec(compile(class_fence, "course/chapters/03.md:CausalAttention", "exec"), class_namespace)
with patch.dict(class_namespace, manual_attention=Mock(wraps=attention.manual_attention)):
    concise_output = class_namespace["CausalAttention"](width=8)(torch.randn(1, 4, 8))
    concise_calls = class_namespace["manual_attention"].call_count
assert concise_calls == 1 and list(concise_output.shape) == [1, 4, 8]

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-6/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": str(v.dtype), "threads": str(torch.get_num_threads())},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "original_fence_count": len(fences),
    "original_stdout": captured.getvalue(),
    "allowed": record_tensor(allowed[0, 0]),
    "weights": record_tensor(weights[0, 0]),
    "out": record_tensor(out[0, 0]),
    "changed_value": record_tensor(changed[0, 0, 3]),
    "changed_out": record_tensor(other[0, 0]),
    "earlier_outputs_exactly_equal": torch.equal(out[:, :, :3], other[:, :, :3]),
    "unmasked_weights": record_tensor(all_weights[0, 0]),
    "unmasked_out": record_tensor(all_out[0, 0]),
    "unmasked_changed_out": record_tensor(all_other[0, 0]),
    "unmasked_original_assertion_failed": assertion_failed,
    "empty_weights": record_tensor(empty_weights[0, 0]),
    "empty_out": record_tensor(empty_out[0, 0]),
    "empty_outputs_finite": bool(torch.isfinite(empty_out).all()),
    "unguarded_all_negative_infinity_softmax_all_nan": bool(torch.isnan(raw_empty_softmax).all()),
    "zero_fill_weights": record_tensor(zero_fill_weights[0, 0]),
    "excerpt_matches_repository_ast": True,
    "repository_matches_excerpt_outputs": True,
    "left_pad_allowed": record_tensor(pad_allowed[0, 0]),
    "left_pad_out": record_tensor(pad_out[0, 0]),
    "segment_allowed": record_tensor(segment_allowed[0, 0]),
    "combined_allowed": record_tensor(combined[0, 0]),
    "library_manual_attention_calls": manual_spy.call_count,
    "library_attention_mask_calls": mask_spy.call_count,
    "library_parameters_unchanged": True,
    "concise_class_manual_attention_calls": concise_calls,
    "shift_pairs": list(zip(list("貓看狗。")[:-1], list("貓看狗。")[1:], strict=True)),
    "result": "全部斷言通過；移除因果遮罩時，原本的前段不變斷言如預期失敗。"
}, ensure_ascii=False, indent=2, allow_nan=False))
