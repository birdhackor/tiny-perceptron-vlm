"""Bounded CPU checks for 5.3; use exact original fence for its suggested changes."""
import contextlib
import copy
import hashlib
import importlib
import io
import json
import math
import platform
import sys
from decimal import Decimal
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
result = {"environment": {"python": sys.version, "executable": sys.executable,
          "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
          "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
          "device": "cpu", "platform": platform.platform(), "threads": str(torch.get_num_threads())}}
original = (HERE / "original/fence-1.py").read_bytes()
result["fence_changes"] = []
for coefficient, expected_velocity, expected_position in [
    ("0.9", ["1", "1.9", ".71"], ["-.1", "-.29", "-.361"]),
    ("0", ["1", "1", "-1"], ["-.1", "-.2", "-.1"]),
    ("0.5", ["1", "1.5", "-.25"], ["-.1", "-.25", "-.225"]),
]:
    code = original.replace(b"0.9 * velocity", coefficient.encode() + b" * velocity")
    name = "fence-mu-" + coefficient.replace(".", "_") + ".py"
    (HERE / name).write_bytes(code)
    namespace = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, str(HERE / name), "exec"), namespace)
    velocity = Decimal(0)
    position = Decimal(0)
    exact = []
    for index, gradient in enumerate([Decimal(1), Decimal(1), Decimal(-1)]):
        velocity = Decimal(coefficient) * velocity + gradient
        position -= Decimal("0.1") * velocity
        assert velocity == Decimal(expected_velocity[index])
        assert position == Decimal(expected_position[index])
        exact.append({"gradient": str(gradient), "velocity": str(velocity), "position": str(position)})
    assert math.isclose(namespace["position"], float(position), abs_tol=1e-12)
    assert math.isclose(namespace["velocity"], float(velocity), abs_tol=1e-12)
    result["fence_changes"].append({"mu": coefficient, "code": name, "stdout": out.getvalue(),
                                  "exact_decimal_check": exact, "status": "pass"})

# Official SGD must realize the same buffer and actual update, not merely report gradients.
w = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
opt = torch.optim.SGD([w], lr=0.1, momentum=0.9, dampening=0.0,
                      weight_decay=0.0, nesterov=False, foreach=False)
result["sgd_sequence"] = []
for gradient, expected_v, expected_w in zip([1., 1., -1.], [1., 1.9, .71], [-.1, -.29, -.361], strict=True):
    w.grad = torch.tensor(gradient, dtype=torch.float64)
    opt.step()
    velocity = opt.state[w]["momentum_buffer"].item()
    assert math.isclose(w.item(), expected_w, abs_tol=1e-12)
    assert math.isclose(velocity, expected_v, abs_tol=1e-12)
    result["sgd_sequence"].append({"gradient": gradient, "buffer": velocity, "w": w.item()})

# Fresh backwards on the same unchanged parameter accumulate, yet do not update w or v.
p = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
acc_opt = torch.optim.SGD([p], lr=0.1, momentum=0.9, foreach=False)
(p.square()).backward()
first = p.grad.item()
(p.square()).backward()
second = p.grad.item()
assert [first, second, p.item()] == [4., 8., 2.]
assert not acc_opt.state
acc_opt.step()
after_step = {"grad": p.grad.item(), "buffer": acc_opt.state[p]["momentum_buffer"].item(), "w": p.item()}
acc_opt.zero_grad(set_to_none=True)
assert p.grad is None
assert acc_opt.state[p]["momentum_buffer"].item() == 8.
(p.square()).backward()
next_gradient = p.grad.item()
acc_opt.step()
next_buffer = acc_opt.state[p]["momentum_buffer"].item()
assert math.isclose(next_buffer, 0.9 * 8 + next_gradient, abs_tol=1e-12)
result["accumulation_vs_momentum"] = {"before_step": {"first_grad": first, "second_grad": second, "w": 2., "momentum_state": "empty"},
    "after_step": after_step, "after_zero_grad": {"grad": None, "buffer": 8.},
    "next_gradient": next_gradient, "next_buffer": next_buffer, "next_w": p.item()}

# Momentum is one tensor-shaped state per parameter; its entries evolve elementwise.
pair = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
pair_opt = torch.optim.SGD([pair], lr=.1, momentum=.9, foreach=False)
for grad in [[1., -2.], [1., -2.]]:
    pair.grad = torch.tensor(grad, dtype=torch.float64)
    pair_opt.step()
assert torch.allclose(pair_opt.state[pair]["momentum_buffer"], torch.tensor([1.9, -3.8], dtype=torch.float64), atol=1e-12, rtol=0)
result["per_entry_buffer"] = {"param_shape": list(pair.shape), "buffer": pair_opt.state[pair]["momentum_buffer"].tolist(), "w": pair.detach().tolist()}

# Save only small in-memory states: no model checkpoint, weights or data download.
saved_w = w.detach().clone()
saved_state = copy.deepcopy(opt.state_dict())
restored = torch.nn.Parameter(saved_w.clone())
restored_opt = torch.optim.SGD([restored], lr=.1, momentum=.9, foreach=False)
restored_opt.load_state_dict(saved_state)
without_history = torch.nn.Parameter(saved_w.clone())
without_opt = torch.optim.SGD([without_history], lr=.1, momentum=.9, foreach=False)
for parameter, optimizer in [(w, opt), (restored, restored_opt), (without_history, without_opt)]:
    parameter.grad = torch.tensor(-1., dtype=torch.float64)
    optimizer.step()
assert torch.equal(w, restored)
assert not torch.equal(w, without_history)
result["resume"] = {"saved_w": saved_w.item(), "with_history": w.item(), "restored_history": restored.item(),
                    "weights_only_fresh_optimizer": without_history.item(), "saved_buffer": saved_state["state"][0]["momentum_buffer"].item()}

# An actual smooth convex loss can rise after crossing its minimum with retained velocity.
q = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
q_opt = torch.optim.SGD([q], lr=.5, momentum=.9, foreach=False)
result["quadratic_overshoot"] = []
for step in range(3):
    q_opt.zero_grad()
    before = .5 * q.square()
    before.backward()
    gradient = q.grad.item()
    q_opt.step()
    result["quadratic_overshoot"].append({"step": step+1, "before_loss": before.item(), "gradient": gradient,
        "after_w": q.item(), "after_loss": (.5*q.square()).item(), "buffer": q_opt.state[q]["momentum_buffer"].item()})
assert result["quadratic_overshoot"][2]["gradient"] < 0
assert result["quadratic_overshoot"][2]["buffer"] > 0
assert result["quadratic_overshoot"][2]["after_loss"] > result["quadratic_overshoot"][2]["before_loss"]

# Unrolled history weights are mu**age, for 0 <= mu < 1; illustrate equal prior history.
result["memory"] = {"history_formula": "v_t = sum_{i=1}^t mu**(t-i) * g_i, v_0=0",
                    "age10_weight": {str(mu): mu**10 for mu in [.5, .9, .99]}, "reversal_after_ten_positive_gradients": {}}
for mu in [.5, .9, .99]:
    v = 0.
    for _ in range(10):
        v = mu*v + 1.
    for reverse_steps in range(1, 101):
        v = mu*v - 1.
        if v < 0:
            break
    result["memory"]["reversal_after_ten_positive_gradients"][str(mu)] = reverse_steps

# Verify downloaded official pinned source is byte-for-byte the installed API implementation.
result["installed_source_matches"] = []
for module_name, snapshot in [("torch.optim.sgd", "sgd-installed-commit.py"),
                              ("torch.optim.optimizer", "optimizer-installed-commit.py"),
                              ("torch.autograd", "autograd-init-installed-commit.py")]:
    path = Path(importlib.import_module(module_name).__file__)
    installed_raw = path.read_bytes()
    official_raw = (HERE/"sources"/snapshot).read_bytes()
    assert installed_raw == official_raw
    result["installed_source_matches"].append({"module": module_name, "installed_path": str(path),
          "snapshot": "sources/"+snapshot, "sha256": hashlib.sha256(installed_raw).hexdigest(), "equal": True})

result["scope"] = "Three-step scalar illustrations and bounded tensor optimizer checks; no model training, empirical performance or quality judgment on the hand-chosen positions."
(HERE/"probe-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
