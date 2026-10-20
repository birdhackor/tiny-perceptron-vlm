"""核對 softmax、符號分布、標準差與注意力縮放。"""

import hashlib
import inspect
import itertools
import json
import math
import platform
import sys
from fractions import Fraction
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.attention import manual_attention

torch.set_default_device("cpu")
torch.set_num_threads(1)


def moment(values):
    mean = sum(values, Fraction()) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return {"mean": str(mean), "variance": str(variance), "std": math.sqrt(variance)}


result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-5/verify_claims.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": "torch.float32", "seed": "42", "threads": str(torch.get_num_threads())},
}
softmax = {}
for gap in (1, 8):
    actual = torch.tensor([gap, 0.0], dtype=torch.float64).softmax(0)
    expected = [math.exp(gap) / (math.exp(gap) + 1), 1 / (math.exp(gap) + 1)]
    assert max(abs(a - b) for a, b in zip(actual.tolist(), expected)) < 1e-15
    softmax[str(gap)] = actual.tolist()
result["softmax"] = softmax
result["sign_moments"] = moment([Fraction(-1), Fraction(1)])
pairs = list(itertools.product((-1, 1), repeat=2))
result["two_sign_sums"] = {"outcomes": [x + y for x, y in pairs], **moment([Fraction(x + y) for x, y in pairs])}
result["sign_products"] = {"outcomes": [x * y for x, y in pairs], **moment([Fraction(x * y) for x, y in pairs])}
result["exact_binomial_moments"] = []
for d in (4, 64):
    distribution = [(2 * j - d, Fraction(math.comb(d, j), 2**d)) for j in range(d + 1)]
    mean = sum(x * p for x, p in distribution)
    variance = sum((x - mean) ** 2 * p for x, p in distribution)
    assert mean == 0 and variance == d
    result["exact_binomial_moments"].append({"d": d, "mean": str(mean), "variance": str(variance), "divisor": math.sqrt(d), "scaled_variance": str(variance / d)})


def sample():
    torch.manual_seed(42)
    batches = []
    for d in (4, 64):
        q = torch.randint(0, 2, (1000, d)).float() * 2 - 1
        k = torch.randint(0, 2, (1000, d)).float() * 2 - 1
        scores = (q * k).sum(dim=-1)
        scaled = scores / math.sqrt(d)
        amplified = (2 * q * 2 * k).sum(dim=-1)
        amplified_scaled = amplified / math.sqrt(d)
        manual_std = torch.sqrt(((scores.double() - scores.double().mean()) ** 2).sum() / 999).item()
        assert set(q.unique().tolist()) == {-1.0, 1.0}
        assert q.dtype == torch.float32 and scores.shape == (1000,)
        assert torch.equal(amplified, 4 * scores)
        assert abs(scores.std().item() - manual_std) < 1e-6
        assert abs(scaled.std().item() - 1) < 0.05
        assert amplified.std().item() == 4 * scores.std().item()
        assert amplified_scaled.std().item() == 4 * scaled.std().item()
        batches.append({"d": d, "q_shape": list(q.shape), "k_shape": list(k.shape), "scores_shape": list(scores.shape), "q_values": q.unique().tolist(), "dtype": str(q.dtype), "sample_count": scores.numel(), "correction": 1, "variance_denominator": 999, "std": scores.std().item(), "scaled_std": scaled.std().item(), "manual_std_float64": manual_std, "item_type": type(scores.std().item()).__name__, "amplified_std": amplified.std().item(), "amplified_scaled_std": amplified_scaled.std().item(), "std_ratio": amplified.std().item() / scores.std().item(), "scaled_std_ratio": amplified_scaled.std().item() / scaled.std().item()})
    return batches


result["samples"] = sample()
result["seed_repeated_same_environment"] = result["samples"] == sample()
assert result["seed_repeated_same_environment"]
possible_scores = torch.tensor([8.0, -8.0] * 497 + [12.0, -12.0, 4.0, -4.0, 0.0, 0.0])
possible_q = torch.ones(1000, 64)
possible_k = -torch.ones_like(possible_q)
for row, score in enumerate(possible_scores.tolist()):
    possible_k[row, :int((score + 64) / 2)] = 1
realized = (possible_q * possible_k).sum(-1)
assert torch.equal(realized, possible_scores)
assert realized.mean().item() == 0
assert realized.square().sum().item() == 999 * 64
assert realized.std().item() == 8
assert (realized / 8).std().item() == 1
result["finite_sample_exact_equality"] = {"sample_count": 1000, "dimension": 64, "allowed_sign_values": [-1, 1], "score_counts": {"8": 497, "-8": 497, "12": 1, "-12": 1, "4": 1, "-4": 1, "0": 2}, "mean": realized.mean().item(), "sum_squared_scores": realized.square().sum().item(), "variance_denominator": 999, "sample_std": realized.std().item(), "scaled_sample_std": (realized / 8).std().item(), "all_scores_realized_by_sign_dot_products": torch.equal(realized, possible_scores), "scope": "這是獨立等機率抽符號模型中具有非零機率的可能樣本，不是聲稱 seed=42 的當次樣本會得到此結果。"}
q = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 10
k = q.flip(-2)
v = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)
allowed = torch.ones(1, 1, 3, 3, dtype=torch.bool)
out, weights = manual_attention(q, k, v, allowed)
expected_weights = (q @ k.transpose(-2, -1) / 2).softmax(-1)
wrong_width_weights = (q @ k.transpose(-2, -1) / math.sqrt(8)).softmax(-1)
assert torch.equal(weights, expected_weights)
assert torch.equal(out, expected_weights @ v)
assert not torch.allclose(weights, wrong_width_weights)
chapter = (ROOT / "course/chapters/03.md").read_text()
result["manual_attention"] = {"q_shape": list(q.shape), "head_width": q.shape[-1], "divisor": math.sqrt(q.shape[-1]), "weights_match_per_head_formula": torch.equal(weights, expected_weights), "output_matches_formula": torch.equal(out, expected_weights @ v), "wrong_total_width_max_weight_difference": (weights - wrong_width_weights).abs().max().item(), "chapter_contains_library_function": inspect.getsource(manual_attention).strip() in chapter, "source_path": "tiny_perceptron/attention.py", "source_sha256": hashlib.sha256((ROOT / "tiny_perceptron/attention.py").read_bytes()).hexdigest(), "scaling_expression": "math.sqrt(q.shape[-1])"}
assert result["manual_attention"]["chapter_contains_library_function"]
temperature = 2.0
logits = torch.tensor([1.0, 0.0])
result["temperature_analogy"] = {"divisor": temperature, "weights": (logits / temperature).softmax(-1).tolist(), "matches_scaled_logits": torch.equal((logits / temperature).softmax(-1), (logits / math.sqrt(4)).softmax(-1))}
assert result["temperature_analogy"]["matches_scaled_logits"]
result["assertions_passed"] = True
print(json.dumps(result, ensure_ascii=False, indent=2))
