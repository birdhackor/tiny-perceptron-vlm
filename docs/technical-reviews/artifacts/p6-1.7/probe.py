"""Fresh bounded CPU verification of current lesson 1.7; no training or downloads."""

import ast
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None
assert not torch.cuda.is_available()
assert sys.flags.optimize == 0

code = (HERE / "fence-1.py").read_bytes()
assert hashlib.sha256(code).hexdigest() == "33cf8f5b278dd38d5abd5ee61a17f42e5c516ae217e8ab5109468e7cf6c2c628"
namespace = {"__name__": "__main__"}
stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    exec(compile(code, "course/chapters/01.md:248-256", "exec"), namespace)
(HERE / "original-stdout.txt").write_text(stdout.getvalue())
(HERE / "original-stderr.txt").write_text(stderr.getvalue())
z, weights, p = (namespace[name] for name in ("z", "weights", "p"))

raw_exp = [math.exp(value) for value in [1.0, 2.0, -1.0]]
raw_sum = math.fsum(raw_exp)
reference = [value / raw_sum for value in raw_exp]
assert all(abs(actual - shown) <= 0.00005 for actual, shown in zip(reference, [0.2595, 0.7054, 0.0351]))
assert all(abs(actual - shown) <= 0.0005 for actual, shown in zip(raw_exp, [2.718, 7.389, 0.368]))
assert abs(raw_sum - 10.475) <= 0.0005
torch.testing.assert_close(p.double(), torch.tensor(reference, dtype=torch.float64), rtol=1e-6, atol=1e-8)

shifted = (z + 100).softmax(dim=0)
builtin = z.softmax(dim=0)
diff = (p - builtin).abs()
allowance = 1e-8 + 1e-5 * builtin.abs()
assert torch.all(diff <= allowance)
assert torch.allclose(p, builtin)
assert torch.equal(shifted, builtin)

# Show why exponentiating these large float32 logits directly is unsafe.
large_raw_weights = (z + 100).exp()
large_raw_probs = large_raw_weights / large_raw_weights.sum()
large_shifted_weights = ((z + 100) - (z + 100).max()).exp()
large_shifted_probs = large_shifted_weights / large_shifted_weights.sum()
assert not torch.isfinite(large_raw_probs).any()
assert torch.isfinite(large_shifted_probs).all()
assert torch.allclose(large_shifted_probs, builtin)

batch = torch.tensor([[1.0, 2.0, -1.0], [3.0, 2.0, -1.0]])
row_probs = batch.softmax(dim=-1)
column_probs = batch.softmax(dim=0)
assert torch.allclose(row_probs.sum(dim=-1), torch.ones(2))
assert torch.allclose(column_probs.sum(dim=0), torch.ones(3))
assert not torch.allclose(column_probs.sum(dim=-1), torch.ones(2))
assert torch.equal(batch.softmax(dim=1), row_probs)

exercise_exp = [math.exp(value) for value in [3.0, 2.0, -1.0]]
exercise_reference = [value / math.fsum(exercise_exp) for value in exercise_exp]
exercise_probs = batch[1].softmax(dim=0)
assert all(abs(actual - shown) <= 0.00005 for actual, shown in zip(exercise_reference, [0.7214, 0.2654, 0.0132]))
torch.testing.assert_close(exercise_probs.double(), torch.tensor(exercise_reference, dtype=torch.float64), rtol=1e-6, atol=1e-8)
assert exercise_probs[1] < p[1]

dog_logits = torch.tensor([1.0, 2.0, 4.0])
dog_probs = dog_logits.softmax(dim=0)
assert dog_logits.argmax().item() == dog_probs.argmax().item() == 2

allclose_near = torch.allclose(torch.tensor([1.0]), torch.tensor([1.0 + 5e-6]))
allclose_far = torch.allclose(torch.tensor([1.0]), torch.tensor([1.0 + 2e-5]))
assert allclose_near is True and allclose_far is False
silent_assert = io.StringIO()
with contextlib.redirect_stdout(silent_assert):
    exec("assert True", {})
assert silent_assert.getvalue() == ""
try:
    exec("assert False", {})
except AssertionError:
    false_assert_raised = True
else:
    false_assert_raised = False
assert false_assert_raised

calls = [ast.unparse(node.func) for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)]
assert not any(name.endswith((".backward", ".step")) for name in calls)
assert not z.requires_grad and z.tolist() == [1.0, 2.0, -1.0]

result = {
    "environment": {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "cuda_build": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "device": str(z.device),
        "dtype": str(z.dtype),
        "platform": platform.platform(),
        "optimize": str(sys.flags.optimize),
        "threads": str(torch.get_num_threads()),
        "offline": {name: os.environ.get(name, "") for name in ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"]},
    },
    "original": {
        "fence_sha256": hashlib.sha256(code).hexdigest(),
        "executed_unchanged": True,
        "stdout": stdout.getvalue(),
        "stdout_line_count": len(stdout.getvalue().splitlines()),
        "stderr": stderr.getvalue(),
        "assert_passed": True,
        "z_after": z.tolist(),
        "requires_grad": z.requires_grad,
        "calls_from_ast": calls,
    },
    "reference_float64": {"e": math.e, "raw_exp": raw_exp, "raw_sum": raw_sum, "probabilities": reference},
    "float32_example": {"weights": weights.tolist(), "weight_sum": weights.sum().item(), "probabilities": p.tolist(), "probability_sum": p.sum().item(), "builtin": builtin.tolist(), "shifted_plus_100": shifted.tolist(), "absolute_difference": diff.tolist(), "allclose_allowance": allowance.tolist()},
    "stability_float32_plus_100": {"unsafe_weights_repr": repr(large_raw_weights), "unsafe_probs_repr": repr(large_raw_probs), "safe_weights": large_shifted_weights.tolist(), "safe_probs": large_shifted_probs.tolist()},
    "batch_axes": {"shape": list(batch.shape), "candidate_dim": -1, "row_probs": row_probs.tolist(), "row_sums": row_probs.sum(dim=-1).tolist(), "wrong_dim0_probs": column_probs.tolist(), "wrong_dim0_column_sums": column_probs.sum(dim=0).tolist(), "wrong_dim0_row_sums": column_probs.sum(dim=-1).tolist()},
    "exercise_cat_3": {"logits": batch[1].tolist(), "reference_float64": exercise_reference, "probabilities_float32": exercise_probs.tolist(), "look_probability_before": p[1].item(), "look_probability_after": exercise_probs[1].item()},
    "wrong_dog_highest": {"logits": dog_logits.tolist(), "probabilities": dog_probs.tolist(), "argmax_candidate_id": dog_probs.argmax().item(), "answer_input_used": False},
    "allclose_and_assert": {"default_rtol": 1e-5, "default_atol": 1e-8, "near_5e_minus6": allclose_near, "far_2e_minus5": allclose_far, "passing_assert_stdout": silent_assert.getvalue(), "false_assert_raised": false_assert_raised},
    "scope": "Original fence plus bounded deterministic CPU variations. No training, parameter update, model download, GPU, or held-out evaluation.",
}
(HERE / "probe-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(stdout.getvalue(), end="")
print("Original source and bounded CPU variations: all assertions passed.")
