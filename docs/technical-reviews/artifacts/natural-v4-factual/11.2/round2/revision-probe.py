"""Small scalar checks for the actual revised 11.2 wording; no training benchmark."""
import contextlib
import io
import json
import math
import platform
import re
from pathlib import Path

import torch
from torch import nn

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
results = {"environment": {
    "python": platform.python_version(), "torch": torch.__version__,
    "commit": torch.version.git_version, "device": "cpu", "dtype": "float64 scalar probes; float32 prerequisite code",
    "threads": "1", "seed": "not applicable; deterministic scalars", "cuda_available": str(torch.cuda.is_available()),
}, "scope": "Exact prerequisite snippets and at most two scalar AdamW updates per independent branch; no quality/speed benchmark or dataset."}

def parameter(value):
    return nn.Parameter(torch.tensor(value, dtype=torch.float64))

def state(opt, p):
    return {name: value.item() for name, value in opt.state[p].items()}

# The flag prevents new leaf accumulation without clearing a gradient computed earlier.
p = parameter(1.)
(p.square()).backward()
p.requires_grad_(False)
u = torch.tensor(3., dtype=torch.float64, requires_grad=True)
(p * u).backward()
assert p.grad.item() == 2 and u.grad.item() == 1
opt = torch.optim.AdamW([p], lr=.001)
opt.step()
assert p.item() < 1
retained = {"requires_grad": p.requires_grad, "retained_grad_after_new_backward": p.grad.item(), "upstream_grad": u.grad.item(), "after_retained_grad_step": p.item(), "state_after_retained_step": state(opt, p)}
opt.zero_grad(set_to_none=True)
before_p, before_state = p.item(), state(opt, p)
opt.step()
assert p.item() == before_p and state(opt, p) == before_state and p.grad is None
retained.update({"grad_after_clear": p.grad, "after_none_step": p.item(), "state_after_none_step": state(opt, p), "none_skips_parameter_and_state": True})
results["retained_then_none"] = retained

# Exclusion also prevents an update, even if an old gradient remains on the excluded tensor.
excluded, included = parameter(1.), parameter(2.)
(excluded.square()).backward()
excluded.requires_grad_(False)
exclude_opt = torch.optim.AdamW([included], lr=.001)
included.square().backward()
exclude_opt.step()
assert excluded.item() == 1 and excluded.grad.item() == 2
results["excluded_from_optimizer"] = {"excluded_requires_grad": excluded.requires_grad, "excluded_grad": excluded.grad.item(), "excluded_value": excluded.item(), "included_value": included.item(), "optimizer_tensor_count": sum(len(g["params"]) for g in exclude_opt.param_groups)}

# Hand-predicted second-step history contribution with g1=1, g2=0 and no decay.
beta1, beta2, lr, eps = .9, .999, .1, 1e-8
m2, v2 = beta1 * (1-beta1), beta2 * (1-beta2)
history_update = lr * (m2 / (1-beta1**2)) / (math.sqrt(v2 / (1-beta2**2)) + eps)
for decay in (0., .2):
    for grad_kind in ("zero", "None"):
        q = parameter(2.)
        hist_opt = torch.optim.AdamW([q], lr=lr, weight_decay=decay, foreach=False)
        q.grad = torch.ones_like(q)
        hist_opt.step()
        first, first_state = q.item(), state(hist_opt, q)
        q.requires_grad_(False)
        hist_opt.zero_grad(set_to_none=(grad_kind == "None"))
        hist_opt.step()
        if grad_kind == "zero":
            expected = first * (1-lr*decay) - history_update
            assert abs(q.item()-expected) < 1e-12 and q.item() != first
            assert math.isclose(state(hist_opt, q)["exp_avg"], m2, abs_tol=1e-15)
            assert math.isclose(state(hist_opt, q)["exp_avg_sq"], v2, abs_tol=1e-15)
            assert state(hist_opt, q)["step"] == 2
        else:
            expected = first
            assert q.item() == first and state(hist_opt, q) == first_state
        results[f"history_decay_{decay}_{grad_kind}"] = {"initial_value": 2., "lr": lr, "weight_decay": decay, "betas": [beta1,beta2], "epsilon": eps, "first_value": first, "first_state": first_state, "grad_kind_after_zero_grad": grad_kind, "requires_grad": q.requires_grad, "second_expected": expected, "second_observed": q.item(), "second_state": state(hist_opt,q), "independent_history_update": history_update, "absolute_error": abs(q.item()-expected)}

def execute(text):
    env, stdout = {}, io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(text, "necessary prerequisite", "exec"), env)
    return env, stdout.getvalue()

for sid in ("5.4", "5.5"):
    source = (base / f"prerequisite-{sid}.md").read_text()
    code = re.findall(r"```python\n(.*?)```", source, re.S)[0]
    env, stdout = execute(code)
    if sid == "5.4":
        assert torch.allclose(env["update"], torch.tensor([.001, .001]), atol=1e-9, rtol=1e-6)
        results["prerequisite_5_4"] = {"stdout": stdout, "m": env["m"].tolist(), "v": env["v"].tolist(), "m_hat": env["m_hat"].tolist(), "v_hat": env["v_hat"].tolist(), "update": env["update"].tolist()}
        neg, negout = execute(code.replace("[1.0, 100.0]", "[1.0, -100.0]"))
        assert torch.allclose(neg["update"], torch.tensor([.001, -.001]), atol=1e-9, rtol=1e-6)
        results["prerequisite_5_4_negative_exercise"] = {"stdout": negout, "update": neg["update"].tolist()}
    else:
        assert abs(env["w"].item()-1.96) < 1e-6
        results["prerequisite_5_5"] = {"stdout": stdout, "w": env["w"].item()}
        no_decay, out = execute(code.replace("weight_decay=0.2", "weight_decay=0").replace("[1.96]", "[2.0]"))
        assert no_decay["w"].item() == 2
        results["prerequisite_5_5_no_decay_exercise"] = {"stdout": out, "w": no_decay["w"].item()}
        none, out = execute(code.replace("w.grad = torch.zeros_like(w)", "w.grad = None").replace("[1.96]", "[2.0]"))
        assert none["w"].item() == 2
        results["prerequisite_5_5_None_exercise"] = {"stdout": out, "w": none["w"].item()}

(base / "revision-probe-result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2))
print("ALL REVISION BOUNDED CPU ASSERTIONS PASSED")
