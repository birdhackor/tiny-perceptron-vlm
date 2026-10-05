"""Independent, bounded CPU checks for lesson 3.5; no model/data downloads."""
from pathlib import Path
import contextlib
import hashlib
import io
import itertools
import json
import math
import platform
import sys

import torch

torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
OUT = Path(__file__).resolve().parent
N = 1000

# Execute exactly the extracted original fence again; capture its raw printout.
code = (OUT / "original/fence-1.py").read_bytes()
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(code, "original/fence-1.py", "exec"), {})

# Independent double-precision exp calculation for the printed softmax examples.
softmax = []
for gap in (1, 8):
    expected = [1 / (1 + math.exp(-gap)), 1 / (1 + math.exp(gap))]
    observed = torch.tensor([gap, 0], dtype=torch.float64).softmax(0).tolist()
    assert max(abs(a - b) for a, b in zip(expected, observed)) < 1e-15
    softmax.append({"gap": gap, "expected": expected, "observed": observed})

# Four equally likely sums form a whole probability distribution (denominator 4).
two_sums = [a + b for a, b in itertools.product((-1, 1), repeat=2)]
mean = sum(two_sums) / 4
population_variance = sum((s - mean) ** 2 for s in two_sums) / 4
assert two_sums == [-2, 0, 0, 2] and mean == 0 and population_variance == 2
four_sums = [sum(signs) for signs in itertools.product((-1, 1), repeat=4)]
assert sum(s * s for s in four_sums) / 16 == 4
assert math.sqrt(4) == 2 and math.sqrt(64) == 8

records = []
for seed in (42, 7):
    torch.manual_seed(seed)
    for d in (4, 64):
        q = torch.randint(0, 2, (N, d)).float() * 2 - 1
        k = torch.randint(0, 2, (N, d)).float() * 2 - 1
        assert q.shape == k.shape == (N, d)
        assert q.dtype == k.dtype == torch.float32
        assert set(q.unique().tolist()) == set(k.unique().tolist()) == {-1.0, 1.0}
        products = q * k
        scores = products.sum(dim=-1)
        scaled = scores / math.sqrt(d)
        assert scores.shape == scaled.shape == (N,)
        values = scores.tolist()
        sample_mean = sum(values) / N
        manual_std = math.sqrt(sum((x - sample_mean) ** 2 for x in values) / (N - 1))
        original_std = scores.std().item()
        scaled_std = scaled.std().item()
        assert abs(original_std - manual_std) < 1e-6
        assert abs(scaled_std - original_std / math.sqrt(d)) < 1e-6
        # 8% is a bounded demonstration tolerance, not a probability guarantee.
        assert abs(original_std / math.sqrt(d) - 1) < 0.08

        doubled_scores = ((q * 2) * (k * 2)).sum(dim=-1)
        doubled_scaled = doubled_scores / math.sqrt(d)
        assert torch.equal(doubled_scores, scores * 4)
        assert torch.allclose(doubled_scaled, scaled * 4, rtol=0, atol=0)
        assert abs(doubled_scores.std().item() - 4 * original_std) < 1e-5
        assert abs(doubled_scaled.std().item() - 4 * scaled_std) < 1e-5
        # Changing q alone gives factor 2; the factor 4 requires changing both.
        assert torch.equal(((q * 2) * k).sum(-1), scores * 2)
        records.append({"seed": seed, "d": d, "pairs": N, "q_k_shape": list(q.shape),
                        "score_shape": list(scores.shape), "std_denominator": N - 1,
                        "population_std_denominator": N,
                        "unscaled_std": original_std, "scaled_std": scaled_std,
                        "manual_sample_std": manual_std,
                        "correction0_std": scores.std(correction=0).item(),
                        "doubled_q_k_std": doubled_scores.std().item(),
                        "doubled_q_k_scaled_std": doubled_scaled.std().item(),
                        "requires_grad": str(scores.requires_grad)})

# Break coordinate independence while preserving each product's mean 0/variance 1.
# Exactly 500 positive and 500 negative rows, each row repeats the same sign.
balanced = torch.tensor([-1.0, 1.0]).repeat(N // 2).reshape(N, 1)
correlated = []
for d in (4, 64):
    q = balanced.repeat(1, d)
    k = torch.ones_like(q)
    scores = (q * k).sum(-1)
    scaled = scores / math.sqrt(d)
    assert scores.mean().item() == 0
    assert abs(scores.std(correction=0).item() - d) < 1e-6
    assert abs(scaled.std(correction=0).item() - math.sqrt(d)) < 1e-6
    correlated.append({"d": d, "pairs": N, "coordinate_variance": 1,
                       "population_unscaled_std": scores.std(correction=0).item(),
                       "population_scaled_std": scaled.std(correction=0).item(),
                       "denominator": N,
                       "scope": "Boundary example: perfectly correlated products, not the original sampling model."})

result = {
    "environment": {"python": platform.python_version(), "python_executable": sys.executable,
                    "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
                    "device": "cpu", "cuda_build": str(torch.version.cuda),
                    "cuda_available": str(torch.cuda.is_available()), "num_threads": str(torch.get_num_threads())},
    "original_fence_sha256": hashlib.sha256(code).hexdigest(),
    "original_stdout": capture.getvalue(), "softmax": softmax,
    "exact_probability_distribution": {"two_sums": two_sums, "denominator": 4,
                                       "mean": mean, "variance": population_variance,
                                       "std": math.sqrt(population_variance),
                                       "four_sum_population_variance": 4, "four_denominator": 16},
    "sampling_and_scaling": records, "correlation_boundary": correlated,
    "tolerances": {"softmax_absolute": 1e-15, "std_absolute": 1e-6,
                   "doubled_std_absolute": 1e-5, "finite_sample_relative_demo": 0.08},
    "assertions": "all passed", "scope": "Small CPU demonstration, no gradients/parameter updates/full attention or training."
}
(OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
