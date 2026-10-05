"""Independent CPU checks of lesson 1.10; no model, training, or downloads."""
import contextlib
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
CODE = (HERE / "original-fence-1.py").read_bytes()
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(CODE, "lesson-1.10-original-fence", "exec"), namespace)
w, u, loss = (namespace[key] for key in ("w", "u", "loss"))
assert w.item() == 3.0 and w.grad.item() == 24.0
assert w.is_leaf and not u.is_leaf and not loss.is_leaf
assert u.grad_fn is not None and loss.grad_fn is not None
original = {"code_sha256": hashlib.sha256(CODE).hexdigest(), "stdout": stdout.getvalue(),
            "w_after_backward": w.item(), "u": u.item(), "L": loss.item(),
            "w_grad": w.grad.item(), "w_shape": list(w.shape),
            "u_shape": list(u.shape), "loss_shape": list(loss.shape),
            "w_leaf": w.is_leaf, "u_leaf": u.is_leaf,
            "u_grad_fn": type(u.grad_fn).__name__, "L_grad_fn": type(loss.grad_fn).__name__,
            "detached_requires_grad": u.detach().requires_grad,
            "detached_grad_fn_is_none": u.detach().grad_fn is None}

# Meaningful exercise: the coefficient and the second stage's operating point both change.
exercise_code = CODE.replace(b"u = 2 * w", b"u = 3 * w").replace(
    b"2 * u.detach().item(), 2", b"2 * u.detach().item(), 3").replace(
    b"w.grad.item() == 24", b"w.grad.item() == 54")
(HERE / "exercise-3w.py").write_bytes(exercise_code)
namespace = {}
exercise_stdout = io.StringIO()
with contextlib.redirect_stdout(exercise_stdout):
    exec(compile(exercise_code, "lesson-1.10-exercise-3w", "exec"), namespace)
assert namespace["w"].item() == 3.0
assert namespace["u"].item() == 9.0 and namespace["loss"].item() == 81.0
assert namespace["w"].grad.item() == 54.0

# Derive directly: L(w)=(c*w)^2=c^2*w^2, so dL/dw=2*c^2*w.
analytic_cases = []
for coefficient, value in [(2.0, 3.0), (3.0, 3.0), (2.0, -3.0), (2.0, 0.0)]:
    param = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    middle = coefficient * param
    objective = middle.square()
    objective.backward()
    expected = 2 * coefficient**2 * value
    assert param.grad.item() == expected
    analytic_cases.append({"coefficient": coefficient, "w": value, "u": middle.item(),
                           "L": objective.item(), "gradient": param.grad.item(),
                           "expected": expected, "tolerance": "exact equality: representable integers"})

# Small-step interpretation. All quantities are abstract scalars: L-units/w-unit.
h, value, coefficient = 0.001, 3.0, 2.0
objective = lambda param: (coefficient * param) ** 2
delta_u = coefficient * h
exact_delta = objective(value + h) - objective(value)
linear_delta = 2 * (coefficient * value) * delta_u
forward_ratio = exact_delta / h
central_difference = (objective(value + h) - objective(value - h)) / (2 * h)
assert abs(exact_delta - 0.024004) < 1e-12
assert abs(linear_delta - 0.024) < 1e-12
assert abs(central_difference - 24.0) < 1e-10
small_step = {"h_w_units": h, "delta_u": delta_u, "exact_delta_L": exact_delta,
              "linear_delta_L": linear_delta, "neglected_quadratic_term": delta_u**2,
              "exact_forward_difference_denominator": h, "exact_forward_ratio": forward_ratio,
              "central_difference_denominator": 2*h, "central_difference": central_difference,
              "analytic_gradient": 24.0,
              "units": "abstract scalar L-units per w-unit, no physical unit assigned",
              "tolerance": "absolute 1e-12 for delta L; 1e-10 for central derivative"}

# Two paths contribute additively: one path contributes 24, the direct square path 6.
param = torch.tensor(3.0, requires_grad=True)
branch_loss = (2 * param).square() + param.square()
branch_loss.backward()
assert branch_loss.item() == 45 and param.grad.item() == 30 and param.item() == 3
branches = {"expression": "L=(2*w)^2+w^2", "w": param.item(), "L": branch_loss.item(),
            "path_gradients": [24, 6], "summed_gradient": param.grad.item()}

result = {"environment": {"python": sys.version, "torch": torch.__version__,
                         "torch_git_version": torch.version.git_version,
                         "device": str(w.device), "dtype_original": str(w.dtype),
                         "num_threads": torch.get_num_threads(), "platform": platform.platform(),
                         "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()},
          "original": original, "exercise_stdout": exercise_stdout.getvalue(),
          "exercise_code_sha256": hashlib.sha256(exercise_code).hexdigest(),
          "analytic_cases": analytic_cases, "small_step": small_step, "two_paths": branches,
          "scope": "scalar chain rule, code API behavior, original and exercise. No optimization, text model, or training."}
(HERE / "bounded-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
