"""CPU checks of numeric examples and the actual generate contract. No training."""
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import generate

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
rows = []
for scores, rounded in [([3., 1., 0.], [0.980, 0.844, 0.629]), ([3., 1., 2.], [0.867, 0.665, 0.506])]:
    z = torch.tensor(scores)
    before = z.clone()
    for temperature, quoted in zip([0.5, 1., 2.], rounded):
        # An independent double-precision scalar exponential calculation.
        exponents = [math.exp(x / temperature) for x in scores]
        denominator = math.fsum(exponents)
        independent = [e / denominator for e in exponents]
        p = (z / temperature).softmax(0)
        error = max(abs(a - b) for a, b in zip(p.tolist(), independent))
        assert error < 1e-7
        assert abs(sum(p.tolist()) - 1.) < 3e-7
        assert abs(independent[0] - quoted) <= 0.0005
        assert z.argmax().item() == (z / temperature).argmax().item() == 0
        assert torch.equal(z, before)
        rows.append({
            "scores": scores, "temperature": temperature,
            "scaled_scores": (z / temperature).tolist(),
            "exp_sum_denominator": denominator,
            "independent_probability": independent, "float32_probability": p.tolist(),
            "float32_max_abs_error": error, "quoted_highest_probability": quoted,
            "float32_normalization_abs_error": abs(sum(p.tolist()) - 1.),
            "quoted_abs_error": abs(independent[0] - quoted),
            "entropy_nats": -math.fsum(v * math.log(v) for v in independent),
            "greedy_id": int(z.argmax()),
        })
for j in range(3):
    assert rows[j+3]["independent_probability"][2] > rows[j]["independent_probability"][2]
for start in [0, 3]:
    assert rows[start]["entropy_nats"] < rows[start+1]["entropy_nats"] < rows[start+2]["entropy_nats"]

tie = torch.tensor([3., 3., 0.])
tie_id = int(tie.argmax())
assert tie_id == 0
zero_p = (torch.tensor([3., 1., 0.]) / 0.).softmax(0)
assert torch.isnan(zero_p).all()
near_zero = (torch.tensor([3., 1., 0.]) / 0.001).softmax(0)
assert torch.equal(near_zero, torch.tensor([1., 0., 0.]))
g = torch.Generator().manual_seed(42)
p = torch.tensor([3., 1., 0.]).softmax(0)
draws = torch.multinomial(p, 4096, replacement=True, generator=g)
counts = torch.bincount(draws, minlength=3).tolist()
assert all(c > 0 for c in counts)
seeded_draws = {
    str(seed): torch.multinomial(p, 12, replacement=True, generator=torch.Generator().manual_seed(seed)).tolist()
    for seed in [42, 7]
}
assert seeded_draws["42"] != seeded_draws["7"]

class PrescribedLogitModel(torch.nn.Module):
    """A bounded adapter to exercise the real helper, not a trained language model."""
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor([3., 1., 0.]))
        self.config = SimpleNamespace(max_length=32)
        self.calls = 0

    def forward(self, ids, cache=None):
        self.calls += 1
        return {"logits": self.scores.reshape(1, 1, 3).expand(ids.shape[0], ids.shape[1], 3), "cache": None}

model = PrescribedLogitModel()
parameter_before = model.scores.detach().clone()
generation = []
for temperature in [0., 0.5, 2.]:
    torch.manual_seed(7)
    result = generate(model, torch.tensor([[0]]), max_new_tokens=8, temperature=temperature, eos_id=99)
    assert result.shape == (1, 9)
    assert torch.equal(model.scores.detach(), parameter_before)
    assert model.scores.grad is None and model.training
    if temperature == 0.:
        assert result.eq(0).all()
    generation.append({"temperature": temperature, "seed": 7, "ids": result.tolist()})
assert generation[0]["ids"] != generation[2]["ids"]
report = {
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
        "torch_git_version": torch.version.git_version, "device": "cpu",
        "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available())},
    "numeric_rows": rows,
    "axis": "dim=0 is the 3 candidate IDs; scalar probabilities are dimensionless",
    "tolerances": {"independent_vs_float32": "max abs < 1e-7", "float32_normalization": "sum abs error < 3e-7 (three terms)", "quoted_3_decimal": "abs <= 0.0005", "parameter_equality": "exact torch.equal"},
    "tie_argmax_id": tie_id, "zero_temperature_softmax": "all NaN; no usable distribution",
    "near_zero_temperature": 0.001, "near_zero_probabilities": near_zero.tolist(),
    "sampling": {"draws": 4096, "replacement": True, "seed": 42, "counts_by_id": counts},
    "seed_only_variation": {"fixed_logits": [3., 1., 0.], "temperature": 1., "draws_per_seed": 12, "ids": seeded_draws},
    "actual_generate": {"module_sha256": hashlib.sha256((ROOT / "tiny_perceptron/model.py").read_bytes()).hexdigest(),
        "runs": generation, "new_tokens_per_run": 8, "forward_calls": model.calls,
        "parameters_unchanged": True, "grad_remains_none": True, "training_mode_restored": True,
        "scope": "Prescribed logits adapter; no trained language quality measured. API temperature<=0 selects a separate greedy branch, never division by zero."},
}
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
