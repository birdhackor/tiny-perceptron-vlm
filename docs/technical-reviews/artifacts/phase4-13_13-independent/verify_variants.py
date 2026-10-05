"""Bounded CPU checks of section 13.13; no data or model loading."""
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch
from tiny_perceptron.posttraining import ppo_clipped_objective

torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)


def check(epsilon):
    ratio = torch.tensor([0.7, 1, 1.3, 0.7, 1, 1.3], requires_grad=True)
    advantage = torch.tensor([1., 1., 1., -1., -1., -1.], requires_grad=True)
    old_log_probability = torch.full((6,), 0.5).log().requires_grad_()
    # Freeze old in constructing new as well, so the contract test is independent.
    new_log_probability = old_log_probability.detach() + ratio.log()
    result = ppo_clipped_objective(new_log_probability, old_log_probability,
                                   advantage, clip_range=epsilon)
    loss = -result["surrogate"].sum()
    loss.backward()
    expected = [0.7, 1., 1 + epsilon, -(1 - epsilon), -1., -1.3]
    expected_gradient = [-1., -1., 0., 0., 1., 1.]
    assert torch.allclose(result["surrogate"], torch.tensor(expected), atol=1e-12, rtol=0)
    assert torch.allclose(ratio.grad, torch.tensor(expected_gradient), atol=1e-12, rtol=0)
    assert old_log_probability.grad is None and advantage.grad is None
    assert torch.allclose(result["policy_loss"], -result["surrogate"].sum() / 6,
                          atol=1e-12, rtol=0)
    return {"epsilon": epsilon, "surrogate": result["surrogate"].tolist(),
            "loss_gradient_wrt_ratio": ratio.grad.tolist(),
            "new_probability": (0.5 * ratio.detach()).tolist(),
            "sample_count": 6, "sum_loss": loss.item(),
            "helper_mean_loss": result["policy_loss"].item(),
            "old_and_advantage_receive_gradient": False}


results = {"cases": [check(0.2), check(0.1)], "tolerance": "absolute 1e-12; float64 CPU"}
ratio = torch.tensor([0.7, 1.3], requires_grad=True)
advantage = torch.tensor([1., -1.])
wrong = ratio.clamp(0.8, 1.2) * advantage
(-wrong.sum()).backward()
results["clamp_without_minimum"] = {"ratio": ratio.detach().tolist(),
                                     "surrogate": wrong.tolist(),
                                     "loss_gradient_wrt_ratio": ratio.grad.tolist()}
assert ratio.grad.tolist() == [0., 0.]

# One parameter controls both samples. The first is already clipped; the second
# is not. A single bounded synthetic update demonstrates the absence of a bound.
theta = torch.tensor(math.log(0.65 / 0.35), requires_grad=True)
probability = theta.sigmoid()
old = torch.tensor([0.5, 0.8]).log()
new = probability.log().expand(2)
result = ppo_clipped_objective(new, old, torch.ones(2))
first_derivative = torch.autograd.grad(-result["surrogate"][0], theta, retain_graph=True)[0]
total_derivative = torch.autograd.grad(-result["surrogate"].sum(), theta)[0]
updated_theta = theta.detach() - total_derivative.detach()
updated_ratio = updated_theta.sigmoid() / 0.5
assert first_derivative.item() == 0 and total_derivative.item() < 0
assert updated_ratio.item() > 1.3
results["shared_parameter_counterexample"] = {
    "first_ratio_before": result["ratio"][0].item(),
    "second_ratio_before": result["ratio"][1].item(),
    "first_sample_loss_derivative": first_derivative.item(),
    "total_loss_derivative": total_derivative.item(),
    "learning_rate": 1., "first_ratio_after_one_synthetic_update": updated_ratio.item(),
    "scope": "Counterexample to a hard ratio constraint; not training or evaluation."}

invalid = []
for label, args, kwargs in [
    ("shape_mismatch", (torch.zeros(2), torch.zeros(1), torch.ones(2)), {}),
    ("empty", (torch.zeros(0), torch.zeros(0), torch.ones(0)), {}),
    ("clip_zero", (torch.zeros(1), torch.zeros(1), torch.ones(1)), {"clip_range": 0}),
    ("clip_one", (torch.zeros(1), torch.zeros(1), torch.ones(1)), {"clip_range": 1}),
]:
    try:
        ppo_clipped_objective(*args, **kwargs)
    except ValueError as error:
        invalid.append({"case": label, "exception": type(error).__name__, "message": str(error)})
    else:
        raise AssertionError(label)
results["helper_contract_errors"] = invalid
results["environment"] = {"python": sys.version, "torch": torch.__version__,
                           "torch_git_version": torch.version.git_version,
                           "device": "cpu", "cuda_available": str(torch.cuda.is_available()),
                           "platform": platform.platform()}
results["implementation_sha256"] = hashlib.sha256(
    Path("tiny_perceptron/posttraining.py").read_bytes()).hexdigest()
print(json.dumps(results, ensure_ascii=False, indent=2))
