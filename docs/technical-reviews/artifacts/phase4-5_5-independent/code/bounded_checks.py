"""Independent, bounded CPU checks of the exact fence and meaningful local variants."""
from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO
from pathlib import Path
import hashlib
import json
import math
import platform
import sys
import torch
from torch import nn

BASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(42)
original = (BASE / "original/fence-1.py").read_bytes()
variants = {
    "original": original,
    "no-decay": original.replace(b"weight_decay=0.2", b"weight_decay=0.0").replace(b"torch.tensor([1.96])", b"torch.tensor([2.0])"),
    "none-gradient": original.replace(b"w.grad = torch.zeros_like(w)", b"w.grad = None").replace(b"torch.tensor([1.96])", b"torch.tensor([2.0])"),
}

def state_record(optimizer, parameter):
    state = optimizer.state.get(parameter, {})
    return {key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()}

result = {"execution_scope": "CPU; 3 exact/locally modified fence runs and fewer than 12 scalar optimizer updates; no model/data download, no training recipe, no prior result/review inputs",
          "fence_runs": [], "checks": {}}
for name, code in variants.items():
    target = BASE / "code" / ("fence-" + name + ".py")
    target.write_bytes(code)
    namespace = {"__name__": "__main__"}
    output = StringIO()
    with redirect_stdout(output):
        exec(compile(code, str(target), "exec"), namespace)
    w = namespace["w"]
    optimizer = namespace["optimizer"]
    result["fence_runs"].append({"name": name, "source": str(target.relative_to(BASE)),
                                  "sha256": hashlib.sha256(code).hexdigest(), "stdout": output.getvalue(),
                                  "value": w.detach().tolist(), "state": state_record(optimizer, w),
                                  "assertion_passed": True})

# Exact decimal arithmetic verifies the intended numbers independently of the printout.
factor = Decimal(1) - Decimal("0.1") * Decimal("0.2")
remaining = Decimal("2") * factor
shrink = Decimal("2") - remaining
assert (factor, remaining, shrink) == (Decimal("0.98"), Decimal("1.96"), Decimal("0.04"))
observed = result["fence_runs"][0]["value"][0]
assert abs(observed - float(remaining)) < 1e-6
result["checks"]["arithmetic"] = {"factor_exact": str(factor), "remaining_exact": str(remaining),
                                      "shrink_exact": str(shrink), "observed_float32": observed,
                                      "absolute_error": abs(observed - float(remaining)),
                                      "review_tolerance": "abs <= 1e-6; original torch.allclose uses rtol=1e-5, atol=1e-8"}

# An actual quadratic task loss has zero derivative at w=2; the added lambda/2*w^2 contributes lambda*w.
p = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
adam_l2 = torch.optim.Adam([p], lr=0.1, weight_decay=0.0)
task_loss = (p - 2.0).square().sum()
task_gradient = torch.autograd.grad(task_loss, p, retain_graph=True)[0]
penalty = (0.2 / 2) * p.square().sum()
(task_loss + penalty).backward()
total_gradient = p.grad.detach().clone()
adam_l2.step()
expected_l2 = 2 - 0.1 * 0.4 / (0.4 + 1e-8)
assert torch.equal(task_gradient, torch.zeros_like(task_gradient))
assert torch.allclose(total_gradient, torch.tensor([0.4], dtype=torch.float64), rtol=0, atol=1e-12)
assert abs(p.item() - expected_l2) < 1e-12
l2_state = state_record(adam_l2, p)
assert abs(l2_state["exp_avg"][0] - 0.04) < 1e-12
assert abs(l2_state["exp_avg_sq"][0] - 0.00016) < 1e-12
assert result["fence_runs"][0]["state"]["exp_avg"] == [0.0]
assert result["fence_runs"][0]["state"]["exp_avg_sq"] == [0.0]
result["checks"]["l2_vs_adamw"] = {"task_loss_formula": "(w-2)^2", "penalty_formula": "(lambda/2)*w^2, lambda=0.2",
                                        "task_gradient": task_gradient.tolist(), "total_gradient": total_gradient.tolist(),
                                        "adam_l2_expected": expected_l2, "adam_l2_observed": p.item(),
                                        "adam_l2_state": l2_state,
                                        "adamw_decoupled_state": result["fence_runs"][0]["state"],
                                        "scope": "One-step counterexample to equivalence; not a generalization benchmark"}

# Plain SGD is a useful control: with matching coefficients the L2 update is multiplicative shrinkage.
p_sgd = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
sgd = torch.optim.SGD([p_sgd], lr=0.1, weight_decay=0.0)
((0.2 / 2) * p_sgd.square().sum()).backward()
sgd.step()
assert abs(p_sgd.item() - 1.96) < 1e-12
result["checks"]["sgd_control"] = {"observed": p_sgd.item(), "expected": 1.96, "absolute_tolerance": 1e-12}

# Decay of a useful parameter can increase task loss: task minimum at w=2, no task-gradient step.
before_loss = (2.0 - 2.0) ** 2
after_loss = (observed - 2.0) ** 2
assert after_loss > before_loss
result["checks"]["task_loss_can_increase"] = {"task_loss_formula": "(w-2)^2", "before": before_loss,
                                                "after": after_loss, "exact_expected_after": 0.0016,
                                                "scope": "Existence of an increase, not a claim about measured validation quality"}

# A fresh optimizer is necessary for 'only decay': with history, zero grad still leaves a momentum step.
warm = []
for next_gradient in ("zero", "none"):
    q = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    opt = torch.optim.AdamW([q], lr=0.1, weight_decay=0.2)
    q.grad = torch.ones_like(q)
    opt.step()
    before = q.item()
    before_state = state_record(opt, q)
    q.grad = torch.zeros_like(q) if next_gradient == "zero" else None
    opt.step()
    after = q.item()
    after_state = state_record(opt, q)
    if next_gradient == "none":
        assert after == before and before_state == after_state
    else:
        predicted_m = 0.9 * before_state["exp_avg"][0]
        predicted_v = 0.999 * before_state["exp_avg_sq"][0]
        predicted_step = 0.1 * (predicted_m / (1 - 0.9 ** 2)) / (math.sqrt(predicted_v / (1 - 0.999 ** 2)) + 1e-8)
        predicted_after = before * 0.98 - predicted_step
        assert abs(after - predicted_after) < 1e-12
        assert abs(after - before * 0.98) > 0.01
    warm.append({"next_gradient": next_gradient, "before": before, "after": after,
                 "before_state": before_state, "after_state": after_state})
result["checks"]["history_boundary"] = {"runs": warm, "scope": "The textbook fence creates a fresh optimizer, so this boundary does not contradict it"}

# Explicit groups permit a weight to decay while bias and normalization scale remain unchanged.
weight, bias, norm_scale = [nn.Parameter(torch.tensor([2.0], dtype=torch.float64)) for _ in range(3)]
grouped = torch.optim.AdamW([{"params": [weight], "weight_decay": 0.2},
                            {"params": [bias, norm_scale], "weight_decay": 0.0}], lr=0.1)
for item in (weight, bias, norm_scale):
    item.grad = torch.zeros_like(item)
grouped.step()
assert abs(weight.item() - 1.96) < 1e-12 and bias.item() == 2 and norm_scale.item() == 2
result["checks"]["explicit_groups"] = {"weight": weight.item(), "bias": bias.item(), "normalization_scale": norm_scale.item(),
                                         "group_weight_decays": [g["weight_decay"] for g in grouped.param_groups],
                                         "scope": "Demonstrates API grouping; exclusion is an explicit policy, not AdamW's automatic default"}

# Cover every API used by the small fence, including the fact that its gradient is assigned, not backward-computed.
t = nn.Parameter(torch.tensor([2.0]))
zeros = torch.zeros_like(t)
detached = t.detach()
assert t.requires_grad and t.is_leaf and t.grad is None
assert zeros.shape == t.shape and zeros.dtype == t.dtype and zeros.device == t.device
assert not zeros.requires_grad and zeros.tolist() == [0.0]
assert not detached.requires_grad and detached.data_ptr() == t.data_ptr()
assert not torch.allclose(torch.tensor([2.0]), torch.tensor([1.96]))
result["checks"]["api_contracts"] = {"tensor_shape": list(t.shape), "tensor_dtype": str(t.dtype),
                                       "parameter_requires_grad": t.requires_grad, "initial_grad_is_none": t.grad is None,
                                       "zeros_like_same_shape_dtype_device": True, "zeros_like_requires_grad": zeros.requires_grad,
                                       "detach_requires_grad": detached.requires_grad, "detach_shares_storage": True,
                                       "allclose_rejects_2_vs_1_96": True,
                                       "fence_action": "Manually assign zero gradient; optimizer.step truly updates parameter; no backward computation or training"}
result["environment"] = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                             "torch_git_commit": torch.version.git_version, "platform": platform.platform(),
                             "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
                             "threads": str(torch.get_num_threads())}
result["all_assertions_passed"] = True
(BASE / "results/bounded-checks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, indent=2, allow_nan=False))
