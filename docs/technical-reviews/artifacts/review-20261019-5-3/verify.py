import io
import json
import math
import platform
from contextlib import redirect_stdout
from fractions import Fraction
from pathlib import Path

import torch


def trajectory(mu):
    m = Fraction(0)
    w = Fraction(0)
    states = []
    for g in [1, 1, -1]:
        m = mu * m + g
        w -= Fraction(1, 10) * m
        states.append({"m": str(m), "w": str(w)})
    return states


def torch_trajectory(mu):
    w = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([w], lr=0.1, momentum=mu, foreach=False)
    states = []
    for g in [1.0, 1.0, -1.0]:
        w.grad = torch.tensor(g, dtype=torch.float64)
        optimizer.step()
        m = optimizer.state[w].get("momentum_buffer")
        states.append({"m": g if m is None else m.item(), "w": w.item()})
    return states


exact = {str(mu): trajectory(mu) for mu in [Fraction(9, 10), Fraction(0), Fraction(1, 2)]}
actual = {str(mu): torch_trajectory(float(mu)) for mu in [Fraction(9, 10), Fraction(0), Fraction(1, 2)]}
for key, states in exact.items():
    for expected, observed in zip(states, actual[key], strict=True):
        for name in ["m", "w"]:
            assert math.isclose(float(Fraction(expected[name])), observed[name], rel_tol=0, abs_tol=1e-15)

bundle = json.loads(Path(__file__).with_name("section-execution.json").read_text())
namespace = {}
captured = io.StringIO()
with redirect_stdout(captured):
    exec(bundle["files"]["fence-1.py"]["content"], namespace)
assert captured.getvalue() == bundle["files"]["stdout.txt"]["content"]
before_round = (namespace["velocity"], namespace["position"])
round(namespace["velocity"], 3)
round(namespace["position"], 3)
assert before_round == (namespace["velocity"], namespace["position"])

w = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9, foreach=False)
(w.square()).backward()
first = w.grad.item()
(w.square()).backward()
second = w.grad.item()
assert (first, second, w.item()) == (4.0, 8.0, 2.0)
optimizer.step()
after_step_grad = w.grad.item()
buffer_before_zero = optimizer.state[w]["momentum_buffer"].item()
optimizer.zero_grad(set_to_none=True)
buffer_after_zero = optimizer.state[w]["momentum_buffer"].item()
assert w.grad is None
assert after_step_grad == buffer_before_zero == buffer_after_zero == 8.0

parameters = [torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64)),
              torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))]
opt = torch.optim.SGD(parameters, lr=0.1, momentum=0.9, foreach=False)
for p, g in zip(parameters, [[1.0, 2.0], [-3.0]], strict=True):
    p.grad = torch.tensor(g, dtype=torch.float64)
opt.step()
buffers = [opt.state[p]["momentum_buffer"].tolist() for p in parameters]
assert buffers == [[1.0, 2.0], [-3.0]]
memory = io.BytesIO()
torch.save({"parameters": [p.detach().clone() for p in parameters], "optimizer": opt.state_dict()}, memory)
memory.seek(0)
saved = torch.load(memory, weights_only=True)
restored_parameters = [torch.nn.Parameter(p.clone()) for p in saved["parameters"]]
reset_parameters = [torch.nn.Parameter(p.clone()) for p in saved["parameters"]]
restored_opt = torch.optim.SGD(restored_parameters, lr=0.1, momentum=0.9, foreach=False)
restored_opt.load_state_dict(saved["optimizer"])
reset_opt = torch.optim.SGD(reset_parameters, lr=0.1, momentum=0.9, foreach=False)
for group, candidate in [(parameters, opt), (restored_parameters, restored_opt), (reset_parameters, reset_opt)]:
    for p in group:
        p.grad = torch.zeros_like(p)
    candidate.step()
assert all(torch.equal(a, b) for a, b in zip(parameters, restored_parameters, strict=True))
assert any(not torch.equal(a, b) for a, b in zip(parameters, reset_parameters, strict=True))

m, w = Fraction(0), Fraction(-1)
overshoot = []
for _ in range(3):
    gradient = w
    m = Fraction(9, 10) * m + gradient
    w -= Fraction(1, 2) * m
    overshoot.append({"gradient": str(gradient), "m": str(m), "w": str(w), "loss": str(w * w / 2)})
assert Fraction(overshoot[-1]["loss"]) > Fraction(overshoot[-2]["loss"])
assert Fraction(overshoot[-1]["gradient"]) > 0 and Fraction(overshoot[-1]["m"]) < 0

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-3/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "device": "cpu", "dtype": "torch.float64"},
    "exact_trajectories": exact,
    "torch_trajectories": actual,
    "section_stdout": captured.getvalue(),
    "rounding_does_not_change_state": True,
    "backward_accumulation": {"first_grad": first, "second_grad": second, "after_step_grad": after_step_grad, "buffer_before_zero_grad": buffer_before_zero, "buffer_after_zero_grad": buffer_after_zero, "grad_after_zero_grad": None},
    "per_coordinate_buffers": buffers,
    "resume": {"restored_next_parameters_equal": True, "omitting_momentum_changes_next_parameters": True},
    "quadratic_overshoot": {"objective": "w*w/2", "initial_w": "-1", "initial_m": "0", "mu": "9/10", "eta": "1/2", "trajectory": overshoot},
    "memory": {"past_gradient_weight": "mu**k", "half_life_mu_half": math.log(0.5) / math.log(0.5), "half_life_mu_point_nine": math.log(0.5) / math.log(0.9)},
    "result": "全部斷言通過；精確分數、浮點結果、梯度累加、動量保存與過衝反例均符合預期。",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
