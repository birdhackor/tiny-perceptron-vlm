"""核對 Adam 算例、歷史效應與狀態接續；所有計算均在 CPU。"""

import copy
import hashlib
import io
import json
import platform
from contextlib import redirect_stdout
from pathlib import Path

import torch


torch.set_default_device("cpu")
root = Path(__file__).resolve().parents[4]
frozen = root / "docs/course-review-20261019/freeze-01/sources/5.4.md"
raw = frozen.read_bytes()
code = raw.decode().split("```python\n", 1)[1].split("```", 1)[0]
results = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-4/verify_adam.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
    },
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "cases": {},
}


def capture_fence(text):
    scope = {"__name__": "__main__"}
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(text, str(frozen), "exec"), scope)
    return {
        "stdout": output.getvalue(),
        "tensors": {name: scope[name].tolist() for name in ["g", "m", "v", "m_hat", "v_hat", "update"]},
        "parameter_created": any(isinstance(value, torch.nn.Parameter) for value in scope.values()),
    }


positive = capture_fence(code)
negative = capture_fence(code.replace("[1.0, 100.0]", "[1.0, -100.0]"))
results["cases"]["frozen_fence"] = positive
results["cases"]["negative_gradient_exercise"] = negative
assert positive["tensors"]["v"] == negative["tensors"]["v"]
assert negative["tensors"]["update"][1] < 0
assert abs(negative["tensors"]["update"][1] + 0.001) < 1e-9
assert not positive["parameter_created"]


def manual_adam(gradients):
    m = torch.zeros(2, dtype=torch.float64)
    v = torch.zeros_like(m)
    output = []
    for step, values in enumerate(gradients, 1):
        gradient = torch.tensor(values, dtype=torch.float64)
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient.square()
        m_hat = m / (1 - 0.9**step)
        v_hat = v / (1 - 0.999**step)
        update = 0.001 * m_hat / (v_hat.sqrt() + 1e-8)
        output.append({"step": step, "g": gradient.tolist(), "m": m.tolist(), "v": v.tolist(), "m_hat": m_hat.tolist(), "v_hat": v_hat.tolist(), "update": update.tolist()})
    return output


gradients = [[1.0, 100.0], [1.0, 0.0], [-1.0, -1.0]]
trace = manual_adam(gradients)
parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
optimizer = torch.optim.Adam([parameter], lr=0.001, foreach=False)
maximum_error = 0.0
for values, expected in zip(gradients, trace):
    before = parameter.detach().clone()
    parameter.grad = torch.tensor(values, dtype=torch.float64)
    optimizer.step()
    actual = before - parameter.detach()
    expected_tensor = torch.tensor(expected["update"], dtype=torch.float64)
    maximum_error = max(maximum_error, (actual - expected_tensor).abs().max().item())
    torch.testing.assert_close(actual, expected_tensor, rtol=1e-12, atol=1e-15)
assert trace[1]["update"][0] != trace[1]["update"][1]
assert trace[2]["g"][0] < 0 < trace[2]["update"][0]
results["cases"]["history_and_official_adam"] = {"trace": trace, "maximum_absolute_update_error": maximum_error, "state_keys": sorted(optimizer.state[parameter])}

momentum = 0.0
average = 0.0
for _ in range(200):
    momentum = 0.9 * momentum + 1.0
    average = 0.9 * average + 0.1
assert abs(momentum - 10) < 1e-7
assert abs(average - 1) < 1e-8
results["cases"]["constant_gradient"] = {"gradient": 1, "updates": 200, "unscaled_momentum": momentum, "adam_first_moment": average, "geometric_limits": [10, 1]}

zero = manual_adam([[0.0, 0.0]])[0]
assert zero["update"] == [0.0, 0.0]
results["cases"]["zero_gradient"] = zero
results["cases"]["sgd_and_curvature"] = {
    "sgd_update": (0.001 * torch.tensor([1.0, 100.0], dtype=torch.float64)).tolist(),
    "same_gradient_different_curvature": [
        {"function": "f(w)=a*w*w/2+100*w", "a": a, "w_before": 0, "gradient_before": 100, "w_after": -0.1, "cost_change": a * 0.1**2 / 2 - 10}
        for a in [100, 10000]
    ],
    "epsilon_decimal": format(1e-8, ".8f"),
    "epsilon_times_one_hundred_million": 1e-8 * 100_000_000,
}

parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
optimizer = torch.optim.Adam([parameter], lr=0.001, foreach=False)
parameter.grad = torch.tensor([1.0, 100.0], dtype=torch.float64)
optimizer.step()
saved_parameter = parameter.detach().clone()
saved_state = copy.deepcopy(optimizer.state_dict())
restored_parameter = torch.nn.Parameter(saved_parameter.clone())
restored_optimizer = torch.optim.Adam([restored_parameter], lr=0.001, foreach=False)
restored_optimizer.load_state_dict(saved_state)
fresh_parameter = torch.nn.Parameter(saved_parameter.clone())
fresh_optimizer = torch.optim.Adam([fresh_parameter], lr=0.001, foreach=False)
for candidate, opt in [(parameter, optimizer), (restored_parameter, restored_optimizer), (fresh_parameter, fresh_optimizer)]:
    candidate.grad = torch.tensor([2.0, -5.0], dtype=torch.float64)
    opt.step()
restored_difference = (parameter - restored_parameter).abs().max().item()
fresh_difference = (parameter - fresh_parameter).abs().max().item()
assert restored_difference == 0.0
assert fresh_difference > 0.0
results["cases"]["optimizer_resume"] = {"saved_state_keys": sorted(saved_state["state"][0]), "saved_step": saved_state["state"][0]["step"].item(), "restored_maximum_parameter_difference": restored_difference, "fresh_optimizer_maximum_parameter_difference": fresh_difference, "scope": "相同參數、超參數及指定梯度；沒有資料抽樣或隨機模型運算。"}
results["result"] = "全部斷言通過"
print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
