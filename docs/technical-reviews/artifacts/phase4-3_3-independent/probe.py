"""Bounded independent checks of section 3.3 axes, toy arithmetic, and Q/K roles."""
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.attention import CausalAttention

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()

namespace = {"__name__": "__main__"}
exec(compile((OUT / "original/fence-1.py").read_bytes(), "original/fence-1.py", "exec"), namespace)
q, k, scores = (namespace[name] for name in ("q", "k", "scores"))
expected = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
assert torch.equal(scores, expected)
assert not any(item.requires_grad for item in (q, k, scores))
assert torch.equal(k.T.T, k)

extended_k = torch.cat((k, torch.tensor([[2., 0.]])), dim=0)
extended_scores = q @ extended_k.T
assert extended_scores.shape == (2, 4)
assert torch.equal(extended_scores, torch.tensor([[1., 0., 1., 2.], [0., 1., 1., 0.]]))

row_weights = scores.softmax(dim=-1)
e = math.exp(1.)
independent_softmax = torch.tensor([[e/(2*e+1), 1/(2*e+1), e/(2*e+1)],
                                    [1/(2*e+1), e/(2*e+1), e/(2*e+1)]])
assert torch.allclose(row_weights, independent_softmax, rtol=0, atol=1e-7)
assert torch.allclose(row_weights.sum(-1), torch.ones(2), rtol=0, atol=1e-7)
column_weights = scores.softmax(dim=0)
assert torch.allclose(column_weights.sum(0), torch.ones(3), rtol=0, atol=1e-7)
assert torch.allclose(column_weights.sum(-1), torch.full((2,), 1.5), rtol=0, atol=1e-7)

mismatch_error = None
try:
    torch.zeros(2, 3) @ k.T
except RuntimeError as error:
    mismatch_error = str(error)
assert mismatch_error is not None

x = torch.tensor([[[1., 2.]]])
layer = CausalAttention(width=2, heads=1)
with torch.no_grad():
    layer.q.weight.copy_(torch.eye(2))
    layer.k.weight.copy_(torch.tensor([[0., 1.], [1., 0.]]))
    projected_q, projected_k = layer.q(x), layer.k(x)
    assert torch.equal(projected_q, torch.tensor([[[1., 2.]]]))
    assert torch.equal(projected_k, torch.tensor([[[2., 1.]]]))
    frozen_k_weight = layer.k.weight.clone()
    layer.q.weight.mul_(2.)
    changed_q, unchanged_k = layer.q(x), layer.k(x)
    assert torch.equal(changed_q, torch.tensor([[[2., 4.]]]))
    assert torch.equal(unchanged_k, projected_k)
    assert torch.equal(layer.k.weight, frozen_k_weight)
assert layer.q.weight is not layer.k.weight
assert layer.q.weight.requires_grad and layer.k.weight.requires_grad

source_paths = {
    "linear.py": "torch/nn/modules/linear.py",
    "tensor-docs.py": "torch/_tensor_docs.py",
    "torch-docs.py": "torch/_torch_docs.py",
    "functional.py": "torch/nn/functional.py",
}
installed_root = Path(torch.__file__).resolve().parent.parent
installed_comparisons = {}
for name, path in source_paths.items():
    local = installed_root / path
    official = OUT / "sources" / name
    same = local.read_bytes() == official.read_bytes()
    installed_comparisons[name] = {
        "local_path": str(local), "official_snapshot": str(official.relative_to(ROOT)),
        "local_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
        "official_sha256": hashlib.sha256(official.read_bytes()).hexdigest(),
        "byte_identical": same,
    }
    assert same

result = {
    "environment": {"python": sys.version, "torch": torch.__version__,
                    "torch_git_version": torch.version.git_version, "device": "cpu",
                    "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
                    "threads": str(torch.get_num_threads()), "dtype": str(q.dtype)},
    "original_scores": scores.tolist(), "original_shapes": {"Q": list(q.shape), "K": list(k.shape),
        "K_transposed": list(k.T.shape), "scores": list(scores.shape)},
    "extended_scores": extended_scores.tolist(), "extended_shape": list(extended_scores.shape),
    "row_softmax": row_weights.tolist(), "row_sums": row_weights.sum(-1).tolist(),
    "column_softmax": column_weights.tolist(), "wrong_axis_row_sums": column_weights.sum(-1).tolist(),
    "softmax_denominator": "each query independently: exp(1)+exp(0)+exp(1)=2e+1; no physical units",
    "mismatch_error": mismatch_error,
    "projection": {"same_X": x.tolist(), "Q_identity": projected_q.tolist(), "K_swap": projected_k.tolist(),
                   "Q_changed_only": changed_q.tolist(), "K_unchanged": unchanged_k.tolist(),
                   "weights_independent_and_trainable": True, "backward_calls": 0, "optimizer_steps": 0},
    "original_example_requires_grad": False,
    "tolerance": "integer inputs/products and shapes exact; float32 softmax absolute tolerance 1e-7, rtol=0",
    "installed_official_source_comparison": installed_comparisons,
    "scope": "Q/K arithmetic and matching table only; separate Linear weights exercised without training; no attention-output, GPU, full recipe, data or model downloads",
    "all_assertions_passed": True,
}
(OUT / "probe-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
