"""Bounded CPU checks of 4.2's arithmetic, derivatives, shapes, and limits."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
records = {}

# Execute the exact textual exercise substitution, keeping its source permanent.
exercise = (ROOT / "exercise.py").read_bytes()
namespace = {"__name__": "__main__"}
exec(compile(exercise, str(ROOT / "exercise.py"), "exec"), namespace)
assert torch.equal(namespace["y"], torch.tensor([1.0, 2.0]))
records["exercise"] = {"y": namespace["y"].detach().tolist(), "grad": namespace["x"].grad.tolist(), "exercise_sha256": hashlib.sha256(exercise).hexdigest()}

# Same axes as the original: a length-2 feature vector for a single position.
x = torch.tensor([1.0, 2.0], requires_grad=True)
before = x.detach().clone()
y = x + 2*x
y.sum().backward()
assert torch.equal(x.detach(), before)
assert x.grad.shape == (2,) and y.shape == (2,)
assert y.sum().shape == ()
assert not y.detach().requires_grad
assert torch.equal(y.detach(), y)
records["original_contract"] = {"input": before.tolist(), "input_after_backward": x.detach().tolist(), "output": y.detach().tolist(), "sum": y.sum().item(), "sum_shape": list(y.sum().shape), "grad": x.grad.tolist(), "input_is_leaf": x.is_leaf, "detached_requires_grad": y.detach().requires_grad, "optimizer_or_parameter_update": False}

# Finite increments on each feature; these are deltas, not a loss or average.
records["finite_increment"] = []
for dtype in (torch.float32, torch.float64):
    base = torch.tensor([1.0, 2.0], dtype=dtype)
    for j in (0, 1):
        step = torch.zeros_like(base)
        step[j] = 0.001
        identity_delta = (base+step)-base
        branch_delta = 2*(base+step)-2*base
        total_delta = ((base+step)+2*(base+step))-(base+2*base)
        expected = torch.zeros_like(base)
        expected[j] = 0.003
        tolerance = 5e-7 if dtype == torch.float32 else 1e-12
        assert torch.allclose(total_delta, expected, rtol=0, atol=tolerance)
        records["finite_increment"].append({"dtype": str(dtype), "feature_index": j, "input_step": 0.001, "identity_delta": identity_delta.tolist(), "branch_delta": branch_delta.tolist(), "output_delta": total_delta.tolist(), "expected_output_delta": expected.tolist(), "absolute_tolerance": tolerance})

# J_y[i,j] = dy_i/dx_j; sum-output derivative is the column sum of J_y.
z = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
A = torch.tensor([[2.0, 0.5], [-1.0, 3.0]], dtype=torch.float64)
J = torch.autograd.functional.jacobian(lambda v: v + A @ v, z)
expected_J = torch.eye(2, dtype=torch.float64) + A
assert torch.equal(J, expected_J)
(z + A@z).sum().backward()
assert torch.equal(z.grad, J.sum(dim=0))
records["mixed_feature_jacobian"] = {"axis_convention": "J[output_feature,input_feature]; grad(sum outputs)=J column sums", "branch_jacobian": A.tolist(), "residual_jacobian": J.tolist(), "identity_plus_branch": expected_J.tolist(), "sum_output_gradient": z.grad.tolist(), "tolerance": "exact equality for binary-representable coefficients"}

records["shape_contract"] = []
for shape in ((1, 1, 2), (2, 3, 4)):
    v = torch.ones(shape, requires_grad=True)
    out = v + 2*v
    out.sum().backward()
    assert tuple(out.shape) == shape and torch.equal(v.grad, torch.full(shape, 3.0))
    records["shape_contract"].append({"input_shape": list(shape), "output_shape": list(out.shape), "all_gradients": 3.0})
try:
    torch.ones(2) + torch.ones(3)
except RuntimeError as error:
    records["incompatible_shape"] = {"shape_pair": [[2], [3]], "rejected": True, "exception": str(error)}
else:
    raise AssertionError("incompatible shape was accepted")

# An identity path does not make the sum injective or guarantee a nonzero gradient.
v = torch.tensor([1.0, 2.0], requires_grad=True)
out = v + (-v)
out.sum().backward()
assert torch.equal(out, torch.zeros(2)) and torch.equal(v.grad, torch.zeros(2))
records["cancellation_boundary"] = {"branch": "f(x)=-x", "output": out.detach().tolist(), "gradient": v.grad.tolist(), "interpretation": "identity contribution remains in algebra, but sum can cancel both information and total derivative; the lesson explicitly denies complete recovery/training guarantees"}

# A small correction value alone does not imply a small derivative.
v = torch.zeros(2, dtype=torch.float64, requires_grad=True)
correction = 1e-8 * torch.sin(1e8*v)
out = v + correction
out.sum().backward()
assert torch.equal(correction, torch.zeros(2)) and torch.equal(v.grad, torch.full((2,), 2.0, dtype=torch.float64))
records["small_value_gradient_boundary"] = {"branch": "1e-8*sin(1e8*x)", "input": v.detach().tolist(), "correction": correction.detach().tolist(), "total_gradient": v.grad.tolist(), "interpretation": "zero branch value at this input need not give identity total sensitivity; the statement refers to the direct contribution"}

environment = {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype_original": "float32", "dtype_jacobian": "float64", "cuda_build": str(torch.version.cuda), "torch_num_threads": str(torch.get_num_threads())}
result = {"command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-4_2-independent/check_residual.py", "environment": environment, "records": records, "all_assertions_passed": True}
(ROOT / "bounded-cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
