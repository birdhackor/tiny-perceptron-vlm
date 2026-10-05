"""Bounded CPU recalculation of 3.2, including its meaningful variations."""
from pathlib import Path
import json
import math
import platform
import sys

import torch
from torch.nn import functional as F

directory = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
environment = {
    "python": sys.version,
    "python_executable": sys.executable,
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu", "dtype": "torch.float32", "threads": "1",
    "platform": platform.platform(), "cwd": str(Path.cwd()),
}
(directory / "probe-environment.json").write_text(json.dumps(environment, indent=2) + "\n")

cases = [
    ("original", [1., 0.], [[1., 0.], [0., 1.], [-1., 0.]], [1., 0., -1.], [0.665, 0.245, 0.090]),
    ("changed_query", [0., 1.], [[1., 0.], [0., 1.], [-1., 0.]], [0., 1., 0.], None),
    ("changed_query_and_third_key", [0., 1.], [[1., 0.], [0., 1.], [-1., 1.]], [0., 1., 1.], [None, 0.422, 0.422]),
    ("first_key_twice_as_long", [1., 0.], [[2., 0.], [0., 1.], [-1., 0.]], [2., 0., -1.], None),
]
results = []
for name, q_data, keys_data, stated_scores, rounded_weights in cases:
    q = torch.tensor(q_data)
    keys = torch.tensor(keys_data)
    scores = q @ keys.T
    weights = scores.softmax(dim=0)
    # An independent Python arithmetic path: no torch.matmul/softmax.
    hand_scores = [sum(a*b for a, b in zip(q_data, key)) for key in keys_data]
    exponentials = [math.exp(score - max(hand_scores)) for score in hand_scores]
    denominator = sum(exponentials)
    hand_weights = [value / denominator for value in exponentials]
    error = max(abs(actual - expected) for actual, expected in zip(weights.tolist(), hand_weights))
    assert scores.tolist() == hand_scores == stated_scores
    assert q.shape == (2,) and keys.shape == (3, 2) and keys.T.shape == (2, 3)
    assert scores.shape == weights.shape == (3,)
    assert torch.equal(scores, torch.matmul(q, keys.T))
    assert error <= 1e-7
    assert bool((weights > 0).all()) and abs(weights.sum().item() - 1.) <= 1e-7
    assert not any(t.requires_grad or t.grad_fn is not None for t in (q, keys, scores, weights))
    rounded_errors = []
    if rounded_weights:
        for actual, rounded in zip(hand_weights, rounded_weights):
            if rounded is not None:
                rounded_errors.append(abs(actual - rounded))
                assert abs(actual - rounded) <= 0.0005
    results.append({
        "case": name, "query": q_data, "keys": keys_data,
        "score_axis": "3 candidate positions", "scores": scores.tolist(),
        "weights": weights.tolist(), "hand_exponentials_after_max_subtraction": exponentials,
        "hand_softmax_denominator": denominator, "hand_weights": hand_weights,
        "max_absolute_float32_error": error,
        "max_rounding_error": max(rounded_errors) if rounded_errors else None,
        "sum_weights": weights.sum().item(), "requires_grad": False,
        "dtype": str(scores.dtype), "shape": list(scores.shape), "status": "pass",
    })

assert results[0]["weights"][0] > results[0]["weights"][1] > results[0]["weights"][2] > 0
assert results[1]["weights"][1] > results[1]["weights"][0] == results[1]["weights"][2]
assert results[2]["weights"][1] == results[2]["weights"][2] > results[2]["weights"][0]
q = torch.tensor([1., 0.])
normal = torch.tensor([1., 0.])
longer = torch.tensor([2., 0.])
assert float(q @ normal) == 1. and float(q @ longer) == 2.
cosine_original = F.cosine_similarity(q, normal, dim=0).item()
cosine_longer = F.cosine_similarity(q, longer, dim=0).item()
assert cosine_original == cosine_longer == 1.
summary = {
    "cases": results, "cosine_check": {
        "query": q.tolist(), "key": normal.tolist(), "longer_key": longer.tolist(),
        "dot_original": 1., "dot_longer": 2.,
        "cosine_original": cosine_original, "cosine_longer": cosine_longer,
    },
    "numeric_policy": {
        "dot_products": "Exact equality for small representable integers.",
        "softmax_float32": "Absolute difference <= 1e-7 versus independently evaluated Python math.exp fractions.",
        "text_rounding": "Three-decimal statements: absolute rounding difference <= 0.0005.",
        "axis": "dim=0 covers all 3 candidate positions of this single-query vector.",
        "units": "Dimensionless manually chosen matching features, scores and proportions; no semantic units.",
        "denominator": "Each case divides each exp(score) by the sum of the 3 candidate exponentials, equivalently after subtracting the maximum.",
    },
    "scope": "One single-query matching/proportion toy. No V readout, feature scaling, causal mask, optimizer, model, semantic test or training.",
    "status": "all four cases and cosine control passed",
}
(directory / "probe-results.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
