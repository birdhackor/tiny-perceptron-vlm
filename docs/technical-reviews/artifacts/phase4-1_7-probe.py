"""Bounded independent CPU checks for lesson 1.7, without training or downloads."""
import json
import math
import platform
import sys

import torch

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()

def real_softmax(values):
    # Independent double-precision scalar calculation, not torch.softmax.
    weights = [math.exp(value) for value in values]
    denominator = math.fsum(weights)
    return weights, denominator, [weight / denominator for weight in weights]

scores = [1.0, 2.0, -1.0]
raw_weights, raw_denominator, expected = real_softmax(scores)
shifted_weights, shifted_denominator, shifted_expected = real_softmax([-1.0, 0.0, -3.0])
exercise_weights, exercise_denominator, exercise_expected = real_softmax([3.0, 2.0, -1.0])

for observed, rounded, tolerance in (
    (math.e, 2.718, 0.0005),
    (raw_denominator, 10.475, 0.0005),
    *[(v, r, 0.0005) for v, r in zip(raw_weights, [2.718, 7.389, 0.368])],
    *[(v, r, 0.0005) for v, r in zip(shifted_weights, [0.368, 1.000, 0.050])],
    *[(v, r, 0.00005) for v, r in zip(expected, [0.2595, 0.7054, 0.0351])],
    *[(v, r, 0.00005) for v, r in zip(exercise_expected, [0.7214, 0.2654, 0.0132])],
):
    assert abs(observed - rounded) <= tolerance
assert max(abs(a - b) for a, b in zip(expected, shifted_expected)) < 1e-12

z = torch.tensor(scores)
original_z = z.clone()
weights = (z - z.max()).exp()
p = weights / weights.sum()
reference = torch.tensor(expected, dtype=torch.float64)
assert torch.allclose(p.to(torch.float64), reference, rtol=1e-6, atol=1e-7)
assert torch.allclose(p, z.softmax(dim=0))
assert torch.allclose((z + 100).softmax(dim=0), z.softmax(dim=0))
assert torch.equal(z, original_z)
assert not z.requires_grad and z.grad is None and not p.requires_grad

large = z + 100
naive_weights = large.exp()
naive_p = naive_weights / naive_weights.sum()
stable_large = large.softmax(dim=0)
assert torch.isinf(naive_weights).all().item()
assert torch.isnan(naive_p).all().item()
assert torch.isfinite(stable_large).all().item()

batch = torch.tensor([scores, [3.0, 2.0, -1.0]])
batch_weights = (batch - batch.max(dim=-1, keepdim=True).values).exp()
batch_manual = batch_weights / batch_weights.sum(dim=-1, keepdim=True)
batch_p = batch.softmax(dim=-1)
assert torch.allclose(batch_manual, batch_p, rtol=1e-6, atol=1e-7)
assert torch.allclose(batch_p.sum(dim=-1), torch.ones(2), rtol=1e-6, atol=1e-7)
assert torch.allclose(batch_p[1].to(torch.float64), torch.tensor(exercise_expected, dtype=torch.float64), rtol=1e-6, atol=1e-7)
assert batch_p[1, 1] < batch_p[0, 1]
global_weights = (batch - batch.max()).exp()
global_p = global_weights / global_weights.sum()
wrong_axis_p = batch.softmax(dim=0)
assert not torch.allclose(global_p.sum(dim=-1), torch.ones(2))
assert not torch.allclose(wrong_axis_p.sum(dim=-1), torch.ones(2))

dog_highest = torch.tensor([1.0, 2.0, 3.0]).softmax(dim=0)
assert dog_highest.argmax().item() == 2

result = {
    "environment": {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": "cpu",
        "cuda_build": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "dtype": str(z.dtype),
        "threads": str(torch.get_num_threads()),
    },
    "units": "Scores, exponential weights and probabilities are dimensionless; three candidates in order cat/look/dog.",
    "axis": "Original [3] vector: dim=0. Bounded variation [2,3]: rows are two questions, last axis is three candidates.",
    "denominators": {"raw_exp_sum": raw_denominator, "shifted_exp_sum": shifted_denominator, "exercise_exp_sum": exercise_denominator},
    "raw_weights_float64": raw_weights,
    "shifted_weights_float64": shifted_weights,
    "expected_probability_float64": expected,
    "exercise_probability_float64": exercise_expected,
    "manual_probability_float32": p.tolist(),
    "manual_sum_float32": p.sum().item(),
    "manual_vs_builtin_max_abs_error": (p - z.softmax(dim=0)).abs().max().item(),
    "shift100_vs_original_max_abs_error": (stable_large - z.softmax(dim=0)).abs().max().item(),
    "naive_shift100_weights_all_inf": torch.isinf(naive_weights).all().item(),
    "naive_shift100_probability_all_nan": torch.isnan(naive_p).all().item(),
    "stable_shift100_probability": stable_large.tolist(),
    "batch_probability": batch_p.tolist(),
    "batch_row_sums": batch_p.sum(dim=-1).tolist(),
    "global_denominator_row_sums": global_p.sum(dim=-1).tolist(),
    "wrong_dim0_row_sums": wrong_axis_p.sum(dim=-1).tolist(),
    "dog_highest_probability": dog_highest.tolist(),
    "input_unchanged": torch.equal(z, original_z),
    "requires_grad": z.requires_grad,
    "grad": None if z.grad is None else z.grad.tolist(),
    "tolerances": {
        "display_3_decimals": "absolute <= 0.0005",
        "display_4_decimals": "absolute <= 0.00005",
        "float64_shift_identity": "absolute < 1e-12",
        "float32_vs_float64_and_batch": "rtol=1e-6, atol=1e-7",
        "original_allclose_defaults": "rtol=1e-5, atol=1e-8",
        "input_unchanged_and_no_grad": "exact",
    },
    "assertions": "All numerical, stability, axis and no-update assertions passed.",
    "scope": "Forward calculation only; no sampled tokens, targets, gradients, parameter updates, training or model-quality measurement.",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
