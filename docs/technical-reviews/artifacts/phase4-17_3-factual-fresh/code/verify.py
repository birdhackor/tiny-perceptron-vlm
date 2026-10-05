"""Bounded CPU verification of lesson 17.3; no training or model/data download."""

import ast
import hashlib
import json
import platform
import runpy
import sys
from fractions import Fraction as F
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_git_version": torch.version.git_version,
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu",
    "threads": str(torch.get_num_threads()),
}
namespace = runpy.run_path(str(BASE / "code/fence-1.py"))
w, restored, x = (namespace[name] for name in ("w", "restored", "x"))
assert w.dtype == restored.dtype == x.dtype == torch.float32
assert tuple(w.shape) == tuple(restored.shape) == tuple(x.shape) == (2, 2)
assert not any(t.requires_grad for t in (w, restored, x))
expected_original = torch.tensor([[0.9, 0.5], [0.5, -1.9]])
expected_restored = torch.tensor([[0.5, 0.0], [-0.5, 1.0]])
expected_compressed = torch.tensor([[0.5, 0.5], [0.5, -1.5]])
expected_difference = torch.tensor([[0.4, 0.0], [0.0, -0.4]])
torch.testing.assert_close(restored, expected_restored, rtol=0, atol=0)
torch.testing.assert_close(namespace["original_output"], expected_original, rtol=0, atol=1e-6)
torch.testing.assert_close(namespace["compressed_output"], expected_compressed, rtol=0, atol=1e-6)
torch.testing.assert_close(namespace["original_output"] - namespace["compressed_output"], expected_difference, rtol=0, atol=1e-6)
assert abs((w-restored).abs().mean().item() - 0.2) < 1e-7

# Independently derive the textbook's decimal arithmetic with rational numbers.
wf = [[F("0.7"), F("0.2")], [F("-0.7"), F("1.2")]]
rf = [[F("0.5"), F(0)], [F("-0.5"), F(1)]]
xf = [[F(1), F(1)], [F(1), F(-1)]]
dwf = [[a-b for a, b in zip(row_w, row_r)] for row_w, row_r in zip(wf, rf)]
exact_outputs = [[sum(v*a for v, a in zip(row_x, row_w)) for row_w in wf] for row_x in xf]
exact_difference = [[sum(v*d for v, d in zip(row_x, row_dw)) for row_dw in dwf] for row_x in xf]
mae_exact = sum(abs(v) for row in dwf for v in row) / 4
assert mae_exact == F(1, 5)
assert exact_difference == [[F(2, 5), F(0)], [F(0), F(-2, 5)]]

variants = {}
for multiplier in (0, -1, 10):
    changed_x = multiplier * x
    difference = changed_x @ w.T - changed_x @ restored.T
    expected = multiplier * expected_difference
    identity = changed_x @ (w-restored).T
    torch.testing.assert_close(difference, expected, rtol=0, atol=1e-5)
    torch.testing.assert_close(difference, identity, rtol=0, atol=1e-5)
    variants[str(multiplier)] = {
        "difference": difference.tolist(),
        "identity": identity.tolist(),
        "maximum_abs_deviation_from_decimal": (difference-expected).abs().max().item(),
        "weight_mae": (w-restored).abs().mean().item(),
    }

# A non-square variant prevents matching square shapes from hiding an axis mistake.
non_square_x = torch.tensor([[1., 1.], [1., -1.], [2., 0.]])
non_square_output = non_square_x @ w.T
assert tuple(non_square_output.shape) == (3, 2)
torch.testing.assert_close(non_square_output, torch.tensor([[0.9, 0.5], [0.5, -1.9], [1.4, -1.4]]), rtol=0, atol=1e-6)

# A general input bound, not the incorrect per-weight bound.
bound = (w-restored).abs().max() * non_square_x.abs().sum(dim=1, keepdim=True)
non_square_difference = non_square_x @ (w-restored).T
assert bool((non_square_difference.abs() <= bound + 1e-6).all())

tree = ast.parse((BASE / "code/fence-1.py").read_bytes())
calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
assert not any("backward" in c or ".step" in c for c in calls)
result = {
    "verification": "all assertions passed",
    "environment": environment,
    "inputs": {
        "fence_sha256": sha(BASE / "code/fence-1.py"),
        "section_sha256": sha(BASE / "originals/section.md"),
        "frozen_chapter_sha256": sha(BASE / "originals/17-frozen.md"),
        "verification_code_sha256": sha(Path(__file__)),
    },
    "original": {
        "shape_axes": "x[2 inputs,2 features] @ w.T[2 features,2 outputs] -> [2 inputs,2 outputs]",
        "weight_difference": (w-restored).tolist(),
        "weight_mae": (w-restored).abs().mean().item(),
        "weight_mae_denominator": 4,
        "original_output": namespace["original_output"].tolist(),
        "compressed_output": namespace["compressed_output"].tolist(),
        "output_difference": (namespace["original_output"]-namespace["compressed_output"]).tolist(),
        "restored_dtype": str(restored.dtype),
        "restored_element_size_bytes": restored.element_size(),
        "gradient_or_update": "none; default requires_grad=False, no backward or optimizer",
    },
    "exact_rational": {
        "weight_difference": [[str(v) for v in row] for row in dwf],
        "outputs": [[str(v) for v in row] for row in exact_outputs],
        "output_difference": [[str(v) for v in row] for row in exact_difference],
        "mae": str(mae_exact),
        "mae_denominator": 4,
        "identity": "D[b,o]=sum_f x[b,f]*(w[o,f]-restored[o,f])",
        "input_bound": "abs(D[b,o])<=max_f abs(delta_w[o,f])*sum_f abs(x[b,f])",
    },
    "variants": variants,
    "non_square": {"x_shape": list(non_square_x.shape), "output_shape": list(non_square_output.shape), "output": non_square_output.tolist()},
    "tolerances": {"original_float32_atol": "1e-6 with rtol=0", "scaled_variant_atol": "1e-5 with rtol=0", "exact_rational": "exact equality"},
}
(BASE / "execution/verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
