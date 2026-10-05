"""Bounded CPU verification of lesson 2.4 and deliberate counterchecks."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
raw = (OUT / "phase4-2_4-original.py").read_bytes()

def run_fence(code):
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "lesson-2.4-original-or-named-variant", "exec"), namespace)
    return namespace, output.getvalue()

def check_tensor(actual, expected):
    reference = torch.tensor(expected, dtype=actual.dtype)
    assert actual.shape == reference.shape
    assert torch.allclose(actual, reference, rtol=0.0, atol=1e-6)
    return float((actual.detach() - reference).abs().max())

original, stdout = run_fence(raw)
errors = [
    check_tensor(original["hidden"], [[-2.0, 2.0], [0.0, 0.0], [2.0, -2.0]]),
    check_tensor(F.relu(original["hidden"]), [[0.0, 2.0], [0.0, 0.0], [2.0, 0.0]]),
    check_tensor(original["linear"], [[0.0], [0.0], [0.0]]),
    check_tensor(original["curved"], [[2.0], [0.0], [2.0]]),
    check_tensor(original["combined"], [[0.0], [0.0], [0.0]]),
]
assert original["a"].bias is None and original["b"].bias is None
assert torch.equal(original["a"].weight, torch.tensor([[1.0], [-1.0]]))
assert torch.equal(original["b"].weight, torch.tensor([[1.0, 1.0]]))
assert all(parameter.grad is None for layer in (original["a"], original["b"]) for parameter in layer.parameters())
assert all(original[name].grad_fn is not None for name in ("hidden", "linear", "curved", "combined"))
assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"backward", "step"} for node in ast.walk(ast.parse(raw)))

variant3 = raw.replace(b"x = torch.tensor([[-2.0], [0.0], [2.0]])", b"x = torch.tensor([[-2.0], [0.0], [3.0]])")
assert variant3 != raw
(OUT / "phase4-2_4-variant-input3.py").write_bytes(variant3)
changed, changed_stdout = run_fence(variant3)
errors.extend([check_tensor(changed["linear"], [[0.0], [0.0], [0.0]]), check_tensor(changed["curved"], [[2.0], [0.0], [3.0]])])

variant_no_relu = variant3.replace(b"curved = b(F.relu(hidden))", b"curved = b(hidden)")
assert variant_no_relu != variant3
(OUT / "phase4-2_4-variant-no-relu.py").write_bytes(variant_no_relu)
removed, removed_stdout = run_fence(variant_no_relu)
errors.append(check_tensor(removed["curved"], [[0.0], [0.0], [0.0]]))

a, b = original["a"], original["b"]
sample = torch.tensor([[-2.0], [-1.5], [-1.0], [0.0], [1.0], [1.5], [2.0], [3.0]])
sample_result = b(F.relu(a(sample)))
errors.append(check_tensor(sample_result, [[2.0], [1.5], [1.0], [0.0], [1.0], [1.5], [2.0], [3.0]]))
negative_slope = float(((sample_result[1] - sample_result[0]) / (sample[1] - sample[0])).detach())
positive_slope = float(((sample_result[5] - sample_result[4]) / (sample[5] - sample[4])).detach())
assert negative_slope == -1.0 and positive_slope == 1.0
singleton = b(F.relu(a(torch.tensor([[3.0]]))))
assert tuple(singleton.shape) == (1, 1) and tuple(singleton.squeeze(-1).shape) == (1,)

first = nn.Linear(2, 3)
second = nn.Linear(3, 2)
with torch.no_grad():
    first.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.0], [0.0, 3.0]]))
    first.bias.copy_(torch.tensor([1.0, -2.0, 0.5]))
    second.weight.copy_(torch.tensor([[2.0, 1.0, 0.0], [0.0, -1.0, 1.0]]))
    second.bias.copy_(torch.tensor([3.0, -4.0]))
inputs = torch.tensor([[1.0, 2.0], [-2.0, 3.0], [0.0, 0.0]])
combined_weight = second.weight @ first.weight
combined_bias = second.weight @ first.bias + second.bias
layered = second(first(inputs))
collapsed = inputs @ combined_weight.T + combined_bias
errors.extend([
    check_tensor(combined_weight, [[1.0, 4.0], [1.0, 3.0]]),
    check_tensor(combined_bias, [3.0, -1.5]),
    check_tensor(layered, [[12.0, 5.5], [13.0, 5.5], [3.0, -1.5]]),
    check_tensor(collapsed, [[12.0, 5.5], [13.0, 5.5], [3.0, -1.5]]),
])
assert torch.allclose(layered, collapsed, rtol=0.0, atol=1e-6)
assert not torch.allclose(layered, inputs @ combined_weight.T, rtol=0.0, atol=1e-6)

report = {
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "cuda_build": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "dtype": "torch.float32",
        "threads": str(torch.get_num_threads()),
    },
    "original_stdout": stdout,
    "original": {
        "x_shape": list(original["x"].shape),
        "a_weight_shape": list(a.weight.shape),
        "b_weight_shape": list(b.weight.shape),
        "hidden_shape": list(original["hidden"].shape),
        "output_shape": list(original["curved"].shape),
        "printed_output_shape": list(original["curved"].squeeze(-1).shape),
        "combined_weight": (b.weight @ a.weight).detach().tolist(),
        "relu_hidden": F.relu(original["hidden"]).detach().tolist(),
        "linear": original["linear"].detach().tolist(),
        "curved": original["curved"].detach().tolist(),
        "parameters_unchanged_after_forward": True,
        "parameter_grads_all_none": True,
        "backward_or_step_calls_in_original": 0,
        "grad_fn_types": {name: type(original[name].grad_fn).__name__ for name in ("hidden", "linear", "curved", "combined")},
    },
    "input3_stdout": changed_stdout,
    "no_relu_stdout": removed_stdout,
    "slope_checks": {"negative_dx": 0.5, "negative_slope": negative_slope, "positive_dx": 0.5, "positive_slope": positive_slope},
    "singleton_batch": {"before_squeeze": list(singleton.shape), "after_squeeze": list(singleton.squeeze(-1).shape)},
    "affine_with_bias": {
        "formula": "W = W2 @ W1; bias = W2 @ bias1 + bias2; Y = X @ W.T + bias",
        "combined_weight": combined_weight.detach().tolist(),
        "combined_bias": combined_bias.detach().tolist(),
        "layered": layered.detach().tolist(),
        "collapsed": collapsed.detach().tolist(),
        "omitting_combined_bias_detected": True,
    },
    "tolerance": {"rtol": 0.0, "atol": 1e-6, "maximum_observed_absolute_error": max(errors)},
    "scope": "forward arithmetic, shapes, API effects, and algebraic counterchecks; no training, backward, dataset, model download, or generalization assessment",
}
(OUT / "phase4-2_4-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
