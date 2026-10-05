"""Bounded independent CPU checks: original fences, tiny variants, raw-result arithmetic.

No training, model/data download, checkpoint load, or existing-model reevaluation.
"""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import math
import sys
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


result = {
    "environment": {
        "python": sys.version,
        "executable": sys.executable,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "cuda_build": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "threads": str(torch.get_num_threads()),
    },
    "scope": "Unmodified fences; bounded 12-row variants and synthetic 17-example routing-method check; arithmetic of original reported measurements. No training or reevaluation of the reported model.",
    "fences": [],
}
for index in (1, 2):
    path = HERE / f"fence-{index}.py"
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(path.read_bytes(), str(path), "exec"), {})
    result["fences"].append({"file": path.name, "sha256": digest(path), "stdout": stream.getvalue()})
    print(f"ORIGINAL FENCE {index}\n{stream.getvalue()}", end="")

rows = []
for advantage in (8.0, 0.1):
    scores = torch.tensor([[advantage, 0.0, 0.0]]).repeat(12, 1)
    probability = scores.softmax(-1)
    chosen = probability.topk(1, dim=-1).indices
    counts = torch.bincount(chosen.flatten(), minlength=3)
    exact = [math.exp(advantage) / (math.exp(advantage) + 2), 1 / (math.exp(advantage) + 2), 1 / (math.exp(advantage) + 2)]
    assert scores.shape == (12, 3) and chosen.shape == (12, 1)
    assert counts.tolist() == [12, 0, 0]
    assert all(abs(a - b) < 2e-7 for a, b in zip(probability.mean(0).tolist(), exact))
    rows.append({"advantage": advantage, "score_shape": list(scores.shape), "chosen_shape": list(chosen.shape), "probability": probability.mean(0).tolist(), "float64_formula": exact, "dispatch_counts": counts.tolist(), "selection_denominator": int(counts.sum()), "dispatch_fraction": (counts / counts.sum()).tolist()})
exercise = torch.tensor([[0.1, 0.2, 0.0]]).repeat(12, 1)
mixed = torch.cat([torch.tensor([[0.1, 0.0, 0.0]]).repeat(6, 1), torch.tensor([[0.0, 0.1, 0.0]]).repeat(6, 1)])
exercise_counts = torch.bincount(exercise.softmax(-1).topk(1, dim=-1).indices.flatten(), minlength=3).tolist()
mixed_counts = torch.bincount(mixed.softmax(-1).topk(1, dim=-1).indices.flatten(), minlength=3).tolist()
assert exercise_counts == [0, 12, 0] and mixed_counts == [6, 6, 0]
result["routing_examples"] = {"original": rows, "changed_second_expert": exercise_counts, "two_groups_six_each": mixed_counts}


def entropy(fractions):
    return sum(-f * math.log(f) if f > 0 else 0.0 for f in fractions)


original = json.loads((HERE / "primary/moe-result-original.json").read_bytes())
measured = original["results"]["variants"]["top1_aux0"]["validation_routing"]
layer = measured["layers"][1]
counts = layer["dispatch_counts"]
fractions = [count / sum(counts) for count in counts]
percentages = [round(100 * f, 2) for f in fractions]
h = entropy(fractions)
normalized = h / math.log(len(counts))
assert counts == [8536, 25630, 4875, 215]
assert sum(counts) == measured["effective_input_tokens"] == layer["dispatch_denominator"] == 39256
assert percentages == [21.74, 65.29, 12.42, 0.55]
assert abs(h - layer["load_entropy_nats"]) < 1e-12
assert abs(normalized - layer["normalized_load_entropy"]) < 1e-12
assert round(normalized, 5) == 0.64755
assert abs(entropy([0.25] * 4) - math.log(4)) < 1e-15
assert entropy([1, 0, 0, 0]) == 0
assert all(digest(HERE / "primary" / filename) == original["code_sha256"][key] for filename, key in [
    ("architecture.py", "scripts/course_experiments/architecture.py"),
    ("modern.py", "tiny_perceptron/modern.py"),
    ("common.py", "scripts/course_experiments/common.py"),
    ("data.py", "tiny_perceptron/data.py"),
])
result["reported_measurement_arithmetic"] = {"raw_result_sha256": digest(HERE / "primary/moe-result-original.json"), "pointer": "/results/variants/top1_aux0/validation_routing/layers/1", "counts": counts, "denominator": sum(counts), "fractions": fractions, "percentages": percentages, "entropy_nats": h, "normalized_entropy": normalized, "rounded_5_decimals": round(normalized, 5), "uniform_entropy_nats": entropy([0.25] * 4), "single_expert_entropy_nats": entropy([1, 0, 0, 0]), "near_zero_contributions": [[f, -f * math.log(f)] for f in (1e-2, 1e-6, 1e-12)]}

# Execute precisely the original measurement function, with a tiny synthetic router.
# Extracting only named AST nodes avoids unrelated result commentary/constants.
namespace = {"torch": torch, "math": math, "IGNORE": -100}
for filename, name in [("data.py", "pad_batch"), ("architecture.py", "_routing")]:
    raw = (HERE / "primary" / filename).read_bytes()
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    selected = ast.Module(body=[node], type_ignores=[])
    exec(compile(selected, f"{filename}:{node.lineno}-{node.end_lineno}", "exec"), namespace)


class SyntheticFFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Identity()

    def forward(self, scores):
        chosen = scores.reshape(-1, 3).softmax(-1).topk(1, dim=-1).indices
        return scores, torch.tensor(0.0), chosen


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(experts=3)
        block = torch.nn.Module()
        block.ffn = SyntheticFFN()
        self.blocks = torch.nn.ModuleList([block])

    def forward(self, tokens, valid=None):
        score_table = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return self.blocks[0].ffn(score_table[tokens])


examples = [(torch.tensor([0, 0]), torch.tensor([1, 1])) for _ in range(8)]
examples += [(torch.tensor([0]), torch.tensor([1])) for _ in range(8)]
examples += [(torch.tensor([1] * 5), torch.tensor([1] * 5))]
with torch.no_grad():
    synthetic = namespace["_routing"](SyntheticModel(), examples, SimpleNamespace(device="cpu"))
p0 = torch.tensor([2.0, 0.0, 0.0]).softmax(0)
p1 = torch.tensor([0.0, 1.0, 0.0]).softmax(0)
batch_aux = [float(3 * p0[0]), float(3 * p1[1])]
expected_aux = (24 * batch_aux[0] + 5 * batch_aux[1]) / 29
assert synthetic["effective_input_tokens"] == 29
assert synthetic["layers"][0]["dispatch_counts"] == [24, 5, 0]
assert abs(synthetic["layers"][0]["mean_batch_auxiliary"] - expected_aux) < 1e-6
expected_probability = (24 * p0.double() + 5 * p1.double()) / 29
assert all(abs(a - b) < 1e-6 for a, b in zip(synthetic["layers"][0]["mean_router_probability"], expected_probability.tolist()))
result["original_routing_method_synthetic_check"] = {"examples": 17, "batches": 2, "valid_inputs_per_batch": [24, 5], "padding_in_first_batch": 8, "batch_auxiliary": batch_aux, "expected_token_weighted_mean": expected_aux, "unweighted_batch_mean": sum(batch_aux) / 2, "observed": synthetic, "parameter_updates": 0}
result["aggregation_counterexample"] = {"four_batches_counts": [[12, 0, 0, 0], [0, 12, 0, 0], [0, 0, 12, 0], [0, 0, 0, 12]], "each_batch_entropy": 0.0, "global_counts": [12, 12, 12, 12], "global_normalized_entropy": entropy([0.25] * 4) / math.log(4)}
(HERE / "cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: v for k, v in result.items() if k not in {"environment", "fences"}}, ensure_ascii=False, indent=2))
print("ALL BOUNDED CPU ASSERTIONS PASSED")
