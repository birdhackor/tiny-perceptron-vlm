"""Independent two-coordinate CPU checks; no data/model download or training."""
import contextlib
import hashlib
import inspect
import io
import json
import math
import platform
import sys
import warnings
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None
assert not torch.cuda.is_available()

environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "threads": str(torch.get_num_threads()),
    "original_dtype": str(torch.get_default_dtype()),
    "finite_difference_dtype": "Python float / IEEE-754 binary64",
}

original = (OUT / "original-fence-1.py").read_bytes()
exercise = original.replace(b"0.5 * w[1].square()", b"2 * w[1].square()").replace(b"[-4.0, 2.0]", b"[-4.0, 8.0]")
assert exercise != original
(OUT / "exercise-coefficient-2.py").write_bytes(exercise)
results = {}
for label, code, costs, grad in (
    ("original", original, [4.0, 2.0, 6.0], [-4.0, 2.0]),
    ("exercise_coefficient_2", exercise, [4.0, 8.0, 12.0], [-4.0, 8.0]),
):
    namespace = {}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code, label, "exec"), namespace)
    w = namespace["w"]
    observed = [namespace[key].item() for key in ("first_cost", "second_cost", "loss")]
    assert observed == costs
    assert torch.equal(w.grad, torch.tensor(grad))
    assert torch.equal(w.detach(), torch.tensor([1.0, 2.0]))
    assert tuple(w.shape) == (2,) == tuple(w.grad.shape)
    assert w.is_leaf and w.grad_fn is None
    assert namespace["loss"].ndim == 0
    results[label] = {"costs": observed, "parameters_after_backward": w.detach().tolist(), "gradient": w.grad.tolist(), "parameter_shape": list(w.shape), "gradient_shape": list(w.grad.shape), "loss_shape": list(namespace["loss"].shape), "parameter_is_leaf": w.is_leaf, "stdout": printed.getvalue(), "code_sha256": hashlib.sha256(code).hexdigest()}

def loss_value(values, coefficient=0.5):
    return (values[0] - 3.0) ** 2 + coefficient * values[1] ** 2

position = [1.0, 2.0]
h = 0.0001
differences = []
for axis, expected in enumerate((-4.0, 2.0)):
    left = position.copy()
    right = position.copy()
    left[axis] -= h
    right[axis] += h
    other = 1 - axis
    assert left[other] == right[other] == position[other]
    derivative = (loss_value(right) - loss_value(left)) / (2.0 * h)
    assert math.isclose(derivative, expected, rel_tol=0.0, abs_tol=1e-8)
    differences.append({"coordinate": axis, "left_parameters": left, "right_parameters": right, "denominator": 2.0 * h, "left_loss": loss_value(left), "right_loss": loss_value(right), "finite_difference": derivative, "analytic": expected, "absolute_error": abs(derivative-expected)})
results["finite_difference"] = {"h": h, "denominator": "2*h, only the chosen coordinate changes", "unit": "loss units per parameter unit; both are dimensionless in this toy polynomial", "absolute_tolerance": 1e-8, "coordinates": differences}

small = 0.001
point_costs = {
    "baseline": loss_value(position),
    "w0_increases": loss_value([1.0 + small, 2.0]),
    "w0_decreases": loss_value([1.0 - small, 2.0]),
    "w1_increases": loss_value([1.0, 2.0 + small]),
    "w1_decreases": loss_value([1.0, 2.0 - small]),
    "gradient_used_as_parameters": loss_value([-4.0, 2.0]),
}
assert point_costs["w0_increases"] < point_costs["baseline"] < point_costs["w0_decreases"]
assert point_costs["w1_decreases"] < point_costs["baseline"] < point_costs["w1_increases"]
assert point_costs["gradient_used_as_parameters"] == 51.0
results["local_directions"] = {"step": small, "costs": point_costs, "scope": "These finite moves check this polynomial near [1,2], not arbitrary losses or large moves."}

w = torch.tensor([1.0, 2.0], requires_grad=True)
first = (w[0] - 3.0).square()
second = 0.5 * w[1].square()
total = first + second
first_branch = torch.autograd.grad(first, w, retain_graph=True)[0]
second_branch = torch.autograd.grad(second, w, retain_graph=True)[0]
assert torch.equal(first_branch, torch.tensor([-4.0, 0.0]))
assert torch.equal(second_branch, torch.tensor([0.0, 2.0]))
assert w.grad is None
total.backward()
assert torch.equal(w.grad, first_branch + second_branch)
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    intermediate_grads = [first.grad, second.grad, total.grad]
assert intermediate_grads == [None, None, None]
detached = w.detach()
assert not detached.requires_grad and detached.grad_fn is None
assert torch.equal(w.detach(), torch.tensor([1.0, 2.0]))
results["graph_routes"] = {"first_branch_gradient": first_branch.tolist(), "second_branch_gradient": second_branch.tolist(), "total_gradient": w.grad.tolist(), "parameter_shape": list(w.shape), "gradient_shape": list(w.grad.shape), "parameter_is_leaf": w.is_leaf, "first_is_leaf": first.is_leaf, "second_is_leaf": second.is_leaf, "first_grad_fn": type(first.grad_fn).__name__, "second_grad_fn": type(second.grad_fn).__name__, "loss_grad_fn": type(total.grad_fn).__name__, "intermediate_grads_after_backward": intermediate_grads, "nonleaf_access_warnings": [str(item.message) for item in captured], "detach_requires_grad": detached.requires_grad, "parameters_after_detach": w.detach().tolist()}

retained_w = torch.tensor([1.0, 2.0], requires_grad=True)
retained_first = (retained_w[0] - 3.0).square()
retained_second = 0.5 * retained_w[1].square()
retained_first.retain_grad()
retained_second.retain_grad()
(retained_first + retained_second).backward()
assert retained_first.grad.item() == retained_second.grad.item() == 1.0
results["retain_grad_variation"] = {"intermediate_gradients": [retained_first.grad.item(), retained_second.grad.item()], "leaf_gradient": retained_w.grad.tolist(), "scope": "Nonleaf .grad can be retained explicitly; ordinary original code does not request this."}

expected = torch.tensor([-4.0, 2.0])
near = expected + torch.tensor([0.0, 0.00001])
far = expected + torch.tensor([0.0, 0.0001])
before = w.detach().clone()
near_result = torch.allclose(near, expected)
far_result = torch.allclose(far, expected)
assert near_result and not far_result
assert torch.equal(w.detach(), before)
assertion_failed = False
try:
    assert torch.allclose(far, expected)
except AssertionError:
    assertion_failed = True
assert assertion_failed
assert torch.equal(w.detach(), before)
results["comparison_and_assert"] = {"default_rtol": 1e-5, "default_atol": 1e-8, "reference_second_element": 2.0, "second_element_bound": 1e-8 + 1e-5 * 2.0, "near_second_difference": (near-expected)[1].item(), "near_result": near_result, "far_second_difference": (far-expected)[1].item(), "far_result": far_result, "mismatch_assert_raises": assertion_failed, "parameters_unchanged": torch.equal(w.detach(), before), "debug": __debug__}

backward_source = inspect.getsource(torch.Tensor.backward)
(OUT / "installed-backward.py").write_text(backward_source)
installed_path = Path(inspect.getsourcefile(torch.Tensor.backward))
installed_receipt = {"installed_path": str(installed_path), "whole_file_sha256": hashlib.sha256(installed_path.read_bytes()).hexdigest(), "symbol": "torch.Tensor.backward", "symbol_sha256": hashlib.sha256(backward_source.encode()).hexdigest(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version)}
(OUT / "installed-source-receipt.json").write_text(json.dumps(installed_receipt, indent=2) + "\n")
output = {"environment": environment, "results": results}
(OUT / "bounded-results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
