"""Bounded CPU check of lesson 14.10 hand examples and YaRN definitions.

No model, training, data download, or pre-existing results are used.
The numeric axis is two candidate positions; angles are radians per token,
L and wavelengths are token distances, and r and gamma are dimensionless.
"""
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
results = {
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "cuda_build": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "device": "cpu",
        "platform": platform.platform(),
    },
    "lesson_fences": {"python": 0, "other": 0, "execution": "No original fence exists; this is an independent math verifier."},
    "axis_units": "softmax dim=-1 is candidate position, not feature; theta radians/token; L tokens; r=L*theta/(2*pi), gamma and c dimensionless",
    "softmax": [],
    "frequency_ramp": [],
}

for scores, scale, rounded_expected in [
    ([1.0, 0.0], 1.0, [0.73, 0.27]),
    ([1.0, 0.0], 2.0, [0.88, 0.12]),
    ([0.0, 0.0], 1.0, [0.5, 0.5]),
    ([0.0, 0.0], 2.0, [0.5, 0.5]),
    ([7.0, 7.0], 2.0, [0.5, 0.5]),
]:
    logits = torch.tensor(scores, dtype=torch.float64) * scale
    observed = logits.softmax(dim=-1)
    shifted = [math.exp(x - max(logits.tolist())) for x in logits.tolist()]
    denominator = sum(shifted)
    independent = torch.tensor([x / denominator for x in shifted], dtype=torch.float64)
    assert torch.allclose(observed, independent, atol=1e-15, rtol=0)
    assert abs(observed.sum().item() - 1.0) <= 1e-15
    assert all(abs(x - y) < 0.005 for x, y in zip(observed.tolist(), rounded_expected))
    if scores[0] == scores[1]:
        assert observed.tolist() == [0.5, 0.5]
    results["softmax"].append({
        "scores": scores,
        "scale_c": scale,
        "scaled_scores": logits.tolist(),
        "observed_weights": observed.tolist(),
        "stable_exp_denominator": denominator,
        "sum": observed.sum().item(),
        "rounded_expected": rounded_expected,
    })

# Bounded variation: positive scaling preserves score order; stronger scale
# concentrates the two-score distribution. It supplies no correctness label.
scales = [0.5, 1.0, 2.0]
positive_variation = []
for c in scales:
    logits = torch.tensor([1.0, 0.0], dtype=torch.float64) * c
    weights = logits.softmax(-1)
    entropy = -(weights * weights.log()).sum().item()
    assert int(weights.argmax()) == 0
    positive_variation.append({"c": c, "weights": weights.tolist(), "entropy_nats": entropy})
assert all(positive_variation[i]["weights"][0] < positive_variation[i+1]["weights"][0] for i in range(2))
assert all(positive_variation[i]["entropy_nats"] > positive_variation[i+1]["entropy_nats"] for i in range(2))
results["positive_scale_variation"] = positive_variation

# Original paper v3 Eqs. (10), (11), and (13). These chosen constants exercise
# the mechanism, rather than claim model-performance measurements.
L, s, alpha, beta = 4096.0, 8.0, 1.0, 32.0
for r, gamma_expected, factor_expected in [
    (0.5, 0.0, 0.125), (1.0, 0.0, 0.125),
    (16.5, 0.5, 0.5625), (32.0, 1.0, 1.0), (64.0, 1.0, 1.0),
]:
    theta = 2 * math.pi * r / L
    recovered_r = L * theta / (2 * math.pi)
    gamma = min(1.0, max(0.0, (recovered_r - alpha) / (beta - alpha)))
    h_theta = (1-gamma) * theta / s + gamma * theta
    assert math.isclose(recovered_r, r, abs_tol=1e-13)
    assert math.isclose(gamma, gamma_expected, abs_tol=1e-15)
    assert math.isclose(h_theta / theta, factor_expected, abs_tol=1e-15)
    assert theta/s <= h_theta <= theta
    assert math.isclose((1-gamma)*theta + gamma*theta, theta, abs_tol=1e-15)
    results["frequency_ramp"].append({
        "L_tokens": L, "s": s, "alpha": alpha, "beta": beta,
        "rotations_in_original_context_r": r, "theta_radians_per_token": theta,
        "gamma": gamma, "new_theta_radians_per_token": h_theta,
        "new_to_original_frequency": h_theta/theta,
    })

# Eq. (14) scales scores by 1/t; Eq. (15) suggests scaling both q and k
# by m=sqrt(1/t) for the specified LLaMA/Llama2 models. The toy c=2 is
# therefore a generic teaching scale, not a universally specified YaRN value.
m = 1 + 0.1 * math.log(s)
c = m * m
q = torch.tensor([1.0, 2.0], dtype=torch.float64)
k = torch.tensor([3.0, 4.0], dtype=torch.float64)
original = (q @ k).item() / math.sqrt(2)
scaled_both = ((m*q) @ (m*k)).item() / math.sqrt(2)
assert math.isclose(scaled_both, c*original, abs_tol=1e-14)
assert c != 2.0
results["temperature_mapping"] = {"extension_s": s, "sqrt_inverse_temperature_m": m, "score_scale_c_inverse_temperature": c, "original_score": original, "scaled_both_qk_score": scaled_both}
results["source_sha256"] = hashlib.sha256((Path(__file__).parent / "frozen-14.10.md").read_bytes()).hexdigest()
results["status"] = "all assertions passed"
print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))
