"""核對權重衰減算例、練習與初始狀態的適用範圍。"""

import json
import platform

import torch
from torch import nn


def require_close(actual, expected, tolerance=2e-6):
    assert abs(actual - expected) <= tolerance, f"實際值 {actual}，預期值 {expected}"


def fresh_step(start, optimizer_class, decay, gradient):
    parameter = nn.Parameter(torch.tensor([start], dtype=torch.float32))
    method = optimizer_class([parameter], lr=0.1, weight_decay=decay)
    parameter.grad = None if gradient is None else torch.full_like(parameter, gradient)
    method.step()
    state = method.state.get(parameter, {})
    return {
        "start": start,
        "optimizer": optimizer_class.__name__,
        "weight_decay": decay,
        "gradient": gradient,
        "parameter": parameter.item(),
        "requires_grad": parameter.requires_grad,
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype),
        "device": str(parameter.device),
        "state": {name: value.tolist() for name, value in state.items()},
        "decoupled_weight_decay": method.param_groups[0]["decoupled_weight_decay"],
    }


cases = []
for start in (2.0, 20.0):
    for method in (torch.optim.Adam, torch.optim.AdamW):
        row = fresh_step(start, method, 0.2, 0.0)
        if method is torch.optim.Adam:
            penalty_gradient = 0.2 * start
            expected = start - 0.1 * penalty_gradient / (abs(penalty_gradient) + 1e-8)
            row["penalty_gradient"] = penalty_gradient
            row["expected_m"] = 0.1 * penalty_gradient
            row["expected_v"] = 0.001 * penalty_gradient**2
            require_close(row["state"]["exp_avg"][0], row["expected_m"])
            require_close(row["state"]["exp_avg_sq"][0], row["expected_v"])
        else:
            expected = start * (1 - 0.1 * 0.2)
            require_close(row["state"]["exp_avg"][0], 0.0, 0.0)
            require_close(row["state"]["exp_avg_sq"][0], 0.0, 0.0)
        row["expected_parameter"] = expected
        require_close(row["parameter"], expected)
        cases.append(row)

exercises = [
    fresh_step(2.0, torch.optim.AdamW, 0.0, 0.0),
    fresh_step(2.0, torch.optim.AdamW, 0.2, None),
]
for row in exercises:
    require_close(row["parameter"], 2.0, 0.0)
assert exercises[1]["state"] == {}, "沒有梯度時不應建立最佳化器狀態"

explicit_l2 = []
for start in (2.0, 20.0):
    parameter = nn.Parameter(torch.tensor([start]))
    method = torch.optim.Adam([parameter], lr=0.1, weight_decay=0.0)
    penalty = 0.1 * parameter.square().sum()
    penalty.backward()
    gradient = parameter.grad.item()
    method.step()
    require_close(gradient, 2 * 0.1 * start)
    reference = next(row for row in cases if row["start"] == start and row["optimizer"] == "Adam")
    require_close(parameter.item(), reference["parameter"], 0.0)
    explicit_l2.append({"start": start, "gradient": gradient, "parameter": parameter.item()})

parameter = nn.Parameter(torch.tensor([2.0]))
method = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.2)
parameter.grad = torch.ones_like(parameter)
method.step()
before = parameter.item()
state_before = {name: value.clone() for name, value in method.state[parameter].items()}
parameter.grad = None
method.step()
after_none = parameter.item()
require_close(after_none, before, 0.0)
assert all(torch.equal(value, method.state[parameter][name]) for name, value in state_before.items())
parameter.grad = torch.zeros_like(parameter)
method.step()
after_zero = parameter.item()
assert abs(after_zero - before * 0.98) > 0.01, "既有動量的額外步幅應可觀察"

parameter = nn.Parameter(torch.tensor([2.0]))
method = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.2)
loss_before = (parameter - 2).square().sum()
loss_before.backward()
method.step()
loss_after = (parameter - 2).square().sum()
require_close(loss_before.item(), 0.0, 0.0)
require_close(loss_after.item(), 0.0016)

print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "device": "cpu"},
    "configuration": {"steps_per_fresh_optimizer": 1, "dtype": "torch.float32", "lr": 0.1,
                      "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False,
                      "maximize": False, "initial_m": 0.0, "initial_v": 0.0,
                      "numeric_absolute_tolerance": 2e-6},
    "fresh_steps": cases,
    "exercises": exercises,
    "explicit_l2": explicit_l2,
    "nonzero_history": {"before": before, "after_none": after_none, "after_zero": after_zero,
                        "pure_decay_reference": before * 0.98},
    "task_loss_can_increase": {"loss_before": loss_before.item(), "loss_after": loss_after.item(),
                               "loss_definition": "(w-2)^2", "initial_w": 2.0},
    "result": "全部核對通過"
}, ensure_ascii=False, indent=2))
