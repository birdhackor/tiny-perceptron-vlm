"""Independent bounded CPU checks for frozen lesson 3.3; no optimizer or training."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
from pathlib import Path

import torch
from torch import nn

from tiny_perceptron.attention import CausalAttention


torch.set_num_threads(1)
torch.manual_seed(0)
root = Path(__file__).resolve().parents[4]
raw = (root / "course/chapters/03.md").read_text(encoding="utf-8")
body = re.search(r"^## 3\.3 .*?(?=^## |\Z)", raw, re.M | re.S)[0]
digest = hashlib.sha256(body.encode()).hexdigest()
assert digest == "1a61f5043696d9e7f2e9921fd44f6b0c1d696e7121b22cf0ae9310f8979f329c"
namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(re.findall(r"```python\n(.*?)```", body, re.S)[0], namespace)
q, k, scores = (namespace[name] for name in ("q", "k", "scores"))
assert scores.tolist() == [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
assert torch.equal(q @ k.T, torch.matmul(q, k.T))
assert torch.equal(k.T.T, k)
assert k.T.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()

extra_k = torch.cat((k, torch.tensor([[2.0, 0.0]])), dim=0)
extra_scores = q @ extra_k.T
assert tuple(extra_scores.shape) == (2, 4)
assert extra_scores.tolist() == [[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 0.0]]

row_weights = scores.double().softmax(dim=-1)
z = 2 * math.e + 1
expected_weights = torch.tensor(
    [[math.e / z, 1 / z, math.e / z], [1 / z, math.e / z, math.e / z]],
    dtype=torch.float64,
)
torch.testing.assert_close(row_weights, expected_weights, atol=1e-15, rtol=1e-15)
torch.testing.assert_close(row_weights.sum(-1), torch.ones(2, dtype=torch.float64))
wrong_axis = scores.double().softmax(dim=0)
assert not torch.allclose(wrong_axis.sum(-1), torch.ones(2, dtype=torch.float64))
extra_weights = extra_scores.double().softmax(dim=-1)
torch.testing.assert_close(extra_weights.sum(-1), torch.ones(2, dtype=torch.float64))

# A change to another query never changes the first query's row normalization.
changed_scores = scores.double().clone()
changed_scores[1] += torch.tensor([2.0, -3.0, 5.0])
assert torch.equal(changed_scores.softmax(-1)[0], row_weights[0])
assert not torch.equal(changed_scores.softmax(0)[0], wrong_axis[0])

q_layer = nn.Linear(2, 2, bias=False, dtype=torch.float64)
k_layer = nn.Linear(2, 2, bias=False, dtype=torch.float64)
with torch.no_grad():
    q_layer.weight.copy_(torch.eye(2, dtype=torch.float64))
    k_layer.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64))
x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
assert q_layer(x).tolist() == [[1.0, 2.0]]
assert k_layer(x).tolist() == [[2.0, 1.0]]
assert q_layer.weight.data_ptr() != k_layer.weight.data_ptr()

model = CausalAttention(width=2, heads=1).double()
with torch.no_grad():
    model.q.weight.copy_(q_layer.weight)
    model.k.weight.copy_(k_layer.weight)
project_q, project_k = model.q(x), model.k(x)
assert project_q.tolist() == [[1.0, 2.0]]
assert project_k.tolist() == [[2.0, 1.0]]
assert model.q.weight.data_ptr() != model.k.weight.data_ptr()

shape_checks = []
for tq, tk, dk in [(1, 4, 3), (2, 3, 2), (4, 2, 5)]:
    q_test = torch.arange(tq * dk, dtype=torch.float64).reshape(tq, dk)
    k_test = torch.arange(tk * dk, dtype=torch.float64).reshape(tk, dk)
    result = q_test @ k_test.T
    manual = [[sum(float(q_test[i, d] * k_test[j, d]) for d in range(dk))
               for j in range(tk)] for i in range(tq)]
    assert result.tolist() == manual
    assert tuple(result.shape) == (tq, tk)
    shape_checks.append({"Tq": tq, "Tk": tk, "Dk": dk, "output": list(result.shape)})
try:
    torch.ones(2, 2) @ torch.ones(3, 4).T
except RuntimeError as exc:
    mismatch_error = str(exc)
else:
    raise AssertionError("Mismatched feature widths should fail")

print(json.dumps({
    "reviewer_task": "/root/repair_tech_3_3",
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "device": "cpu", "dtype_original": str(q.dtype),
                    "dtype_extended": "torch.float64", "threads": "1", "seed": "0"},
    "source_sha256": digest,
    "original_stdout": captured.getvalue(),
    "original_scores": scores.tolist(),
    "exercise_scores": extra_scores.tolist(),
    "row_softmax": row_weights.tolist(),
    "row_sums": row_weights.sum(-1).tolist(),
    "wrong_axis_row_sums": wrong_axis.sum(-1).tolist(),
    "exercise_row_softmax": extra_weights.tolist(),
    "identity_query": project_q.tolist(), "swapped_key": project_k.tolist(),
    "shape_checks": shape_checks, "mismatched_features_error": mismatch_error,
    "assertions": "all passed",
    "scope": "Original lesson cell, exercise, small exact/float64 calculations, and q/k projections only; zero optimizer updates; no training, generation, GPU, or historic experiment rerun."
}, ensure_ascii=False, indent=2))
