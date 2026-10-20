"""核對 PPO 裁切算例、固定資料的梯度與共享參數更新。"""

import ast
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.posttraining import ppo_clipped_objective

torch.set_default_device("cpu")
TOLERANCE = 1e-6


def close(actual, expected):
    torch.testing.assert_close(actual.detach(), torch.tensor(expected, dtype=actual.dtype), rtol=0, atol=TOLERANCE)


cases = []
for epsilon, expected in [(0.2, [0.7, 1, 1.2, -0.8, -1, -1.3]), (0.1, [0.7, 1, 1.1, -0.9, -1, -1.3])]:
    ratio = torch.tensor([0.7, 1, 1.3, 0.7, 1, 1.3], requires_grad=True)
    advantage = torch.tensor([1., 1., 1., -1., -1., -1.])
    old_log = torch.full((6,), 0.5).log()
    new_log = old_log + ratio.log()
    result = ppo_clipped_objective(new_log, old_log, advantage, epsilon)
    sum_grad = torch.autograd.grad(-result["surrogate"].sum(), ratio, retain_graph=True)[0]
    mean_grad = torch.autograd.grad(result["policy_loss"], ratio)[0]
    close(result["surrogate"], expected)
    close(sum_grad, [-1, -1, 0, 0, 1, 1])
    close(mean_grad, [-1 / 6, -1 / 6, 0, 0, 1 / 6, 1 / 6])
    close(result["ratio"], ratio.detach().tolist())
    close(result["clip_fraction"], 4 / 6)
    cases.append({"clip_range": epsilon, "old_probability": old_log.exp().tolist(),
                  "new_probability": new_log.exp().tolist(), "ratio": result["ratio"].tolist(),
                  "unclipped": result["unclipped"].tolist(), "clipped": result["clipped"].tolist(),
                  "surrogate": result["surrogate"].tolist(), "sum_loss_gradient": sum_grad.tolist(),
                  "mean_loss_gradient": mean_grad.tolist(), "policy_loss": result["policy_loss"].item(),
                  "clip_fraction": result["clip_fraction"].item()})

r = torch.tensor(1.3, requires_grad=True)
clamp_only_loss = -r.clamp(0.8, 1.2) * -1
clamp_only_gradient = torch.autograd.grad(clamp_only_loss, r)[0]
minimum_loss = -torch.minimum(r * -1, r.clamp(0.8, 1.2) * -1)
minimum_gradient = torch.autograd.grad(minimum_loss, r)[0]
close(clamp_only_gradient, 0.)
close(minimum_gradient, 1.)

new_log = torch.tensor([-0.9, -0.4], requires_grad=True)
old_log = torch.tensor([-0.8, -0.5], requires_grad=True)
advantage = torch.tensor([1., -1.], requires_grad=True)
detached = ppo_clipped_objective(new_log, old_log, advantage)
detached["policy_loss"].backward()
assert old_log.grad is None and advantage.grad is None
assert new_log.grad is not None

invalid = []
for name, args, kwargs in [
    ("不同形狀", (torch.zeros(2), torch.zeros(1), torch.ones(2)), {}),
    ("優勢形狀不同", (torch.zeros(2), torch.zeros(2), torch.ones(1)), {}),
    ("空輸入", (torch.empty(0), torch.empty(0), torch.empty(0)), {}),
    ("裁切幅度為零", (torch.zeros(2), torch.zeros(2), torch.ones(2)), {"clip_range": 0}),
    ("裁切幅度為一", (torch.zeros(2), torch.zeros(2), torch.ones(2)), {"clip_range": 1}),
    ("負裁切幅度", (torch.zeros(2), torch.zeros(2), torch.ones(2)), {"clip_range": -0.2}),
]:
    try:
        ppo_clipped_objective(*args, **kwargs)
    except ValueError as error:
        invalid.append({"case": name, "exception": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError("無效輸入未遭拒絕")

source = (ROOT / "docs/course-review-20261019/freeze-01/sources/13.13.md").read_text()
blocks = source.split("```python\n")[1:]
excerpt = blocks[1].split("```", 1)[0]
excerpt_node = next(n for n in ast.parse(excerpt).body if isinstance(n, ast.FunctionDef))
actual_node = ast.parse(inspect.getsource(ppo_clipped_objective)).body[0]
assert ast.dump(excerpt_node) == ast.dump(actual_node)

old_theta = torch.tensor(0.)
x = torch.tensor([2., 0.2])
old_prob = torch.sigmoid(old_theta * x)
theta = torch.tensor(torch.logit(torch.tensor(0.65)).item() / 2, requires_grad=True)
prob_before = torch.sigmoid(theta * x)
shared = ppo_clipped_objective(prob_before.log(), old_prob.log(), torch.ones(2))
individual_gradient = [torch.autograd.grad(-v, theta, retain_graph=True)[0].item() for v in shared["surrogate"]]
grad = torch.autograd.grad(-shared["surrogate"].sum(), theta)[0]
learning_rate = 0.5
theta_after = theta.detach() - learning_rate * grad
ratio_after = torch.sigmoid(theta_after * x) / old_prob
assert individual_gradient[0] == 0 and individual_gradient[1] < 0
assert ratio_after[0] > shared["ratio"][0] > 1.2

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-13/verify-objective.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": str(torch.get_default_dtype())},
    "result": "全部斷言通過", "absolute_tolerance": TOLERANCE, "cases": cases,
    "minimum_comparison": {"ratio": 1.3, "advantage": -1, "clamp_only_loss_gradient": clamp_only_gradient.item(),
                           "minimum_loss_gradient": minimum_gradient.item()},
    "detach": {"new_log_gradient": new_log.grad.tolist(), "old_log_gradient": old_log.grad, "advantage_gradient": advantage.grad},
    "invalid_inputs": invalid, "excerpt_ast_equals_repository_function": True,
    "shared_parameter_example": {"model": "p(action|x)=sigmoid(theta*x)", "features": x.tolist(),
        "old_theta": old_theta.item(), "old_probability": old_prob.tolist(), "theta_before": theta.item(),
        "ratio_before": shared["ratio"].tolist(), "advantage": [1, 1], "individual_loss_gradient": individual_gradient,
        "learning_rate": learning_rate, "theta_after": theta_after.item(), "ratio_after": ratio_after.tolist()},
    "repository_source_sha256": hashlib.sha256((ROOT / "tiny_perceptron/posttraining.py").read_bytes()).hexdigest(),
}
print(json.dumps(result, ensure_ascii=False, indent=2))
