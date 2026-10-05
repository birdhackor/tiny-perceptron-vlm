"""Independent 5.16 checks: arithmetic, small CPU tensors, existing JSON/sampling only.

No model is instantiated; no forward pass, gradients, optimizer, data preparation,
checkpoint read, network request or training is performed.
"""
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from tiny_perceptron.data import IGNORE, pad_batch
from tiny_perceptron.model import loss_sum, masked_loss
from scripts.course_experiments.common import records_sha256, text_examples

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")

def show(name, value):
    print(name + " " + json.dumps(value, ensure_ascii=False, sort_keys=True))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

original = (BASE / "original/fence-1.py").read_bytes()
for name, code, expected in [
    ("original", original, {"短答": 90, "長答": 200}),
    ("long_length_1", original.replace(b'"\xe9\x95\xb7\xe7\xad\x94": 20', b'"\xe9\x95\xb7\xe7\xad\x94": 1'), {"短答": 90, "長答": 10}),
    ("long_count_5", original.replace(b'"\xe9\x95\xb7\xe7\xad\x94": 10', b'"\xe9\x95\xb7\xe7\xad\x94": 5'), {"短答": 90, "長答": 100}),
]:
    assert name == "original" or code != original
    namespace = {}
    (BASE / (name + "-executed.py")).write_bytes(code)
    exec(compile(code, str(BASE / (name + "-executed.py")), "exec"), namespace)
    assert namespace["tokens"] == expected
    for key, value in expected.items():
        assert namespace["ratios"][key] == float(Fraction(value, sum(expected.values())))
    show("fence_case", {"case": name, "tokens": namespace["tokens"], "total": namespace["total"], "ratios": namespace["ratios"]})

assert Fraction(90, 100) == Fraction(9, 10)
assert Fraction(180, 20) == 9 and Fraction(20, 180 + 20) == Fraction(1, 10)
show("exact_arithmetic", {"short_record_fraction": "90/100=9/10", "long_token_fraction": "200/290=20/29", "token_target_90_10_record_ratio": "180:1", "exercise_long_count_5": "100/190=10/19"})

# Abstract labels preserve the original example's stipulated valid lengths;
# two leading ignored positions represent input-only text, not a new dataset.
examples = [(torch.zeros(n + 2, dtype=torch.long), torch.tensor([IGNORE, IGNORE] + [0] * n)) for n in [1] * 90 + [20] * 10]
x, y, valid = pad_batch(examples)
assert x.shape == (100, 22) and int((y != IGNORE).sum()) == 290
logits = torch.zeros(100, 22, 2, dtype=torch.float64)
summed, count = loss_sum(logits, y)
mean = masked_loss(logits, y)
assert int(count) == 290
assert math.isclose(float(summed), 290 * math.log(2), abs_tol=1e-10)
assert math.isclose(float(mean), math.log(2), abs_tol=1e-12)
assert torch.equal(mean, F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1), ignore_index=IGNORE))
changed = logits.clone()
changed[y == IGNORE] = torch.tensor([100.0, -100.0], dtype=torch.float64)
assert torch.equal(mean, masked_loss(changed, y))
_, cropped, _ = pad_batch(examples, max_length=12)
assert int((cropped != IGNORE).sum()) == 190
show("masked_cpu", {"padded_shape": list(y.shape), "padded_cells": y.numel(), "valid_targets": int(count), "ignored_targets": int((y == IGNORE).sum()), "sum_nll": float(summed), "mean_nll": float(mean), "ignored_logit_change_equal": True, "cropped_12_cells_valid_targets": 190})

# Long-answer loss values can be smaller despite their greater position count.
different = logits.clone()
different[:90, :, 1] = 2
different[90:, :, 0] = 2
unreduced = F.cross_entropy(different.reshape(-1, 2), y.reshape(-1), ignore_index=IGNORE, reduction="none").reshape(y.shape)
source_sums = [float(unreduced[:90].sum()), float(unreduced[90:].sum())]
assert source_sums[1] / sum(source_sums) < 0.12
source_weights = torch.ones_like(y, dtype=torch.float64)
source_weights[90:] = 1 / 20
effective_weights = source_weights * (y != IGNORE)
assert math.isclose(float(effective_weights[:90].sum()), 90)
assert math.isclose(float(effective_weights[90:].sum()), 10)
assert int((y != IGNORE).sum()) == 290
show("position_vs_loss", {"long_position_share": 200 / 290, "source_nll_sums_short_long": source_sums, "long_nll_share": source_sums[1] / sum(source_sums), "custom_source_weights_short_long": [1, 0.05], "weighted_position_masses_short_long": [90, 10], "unchanged_valid_positions": 290, "note": "Source scaling is an explicit custom illustration, not CrossEntropyLoss.class_weight or an implemented training recipe."})

# Sampling source probability does not force a fixed per-batch composition.
probability = 1 / 181
p_no_long = (1 - probability) ** 16
expected_batch_long_share = sum(math.comb(16, k) * probability**k * (1-probability)**(16-k) * (20*k / (16+19*k)) for k in range(17))
rng = random.Random(42)
small_batches = [Counter(rng.choices(["short"] * 180 + ["long"], k=16)) for _ in range(20)]
assert any(b["long"] == 0 for b in small_batches)
show("small_batch_scope", {"stipulated_record_ratio": "180:1", "long_draw_probability": probability, "ratio_of_expected_token_totals": 20*probability / (1-probability + 20*probability), "probability_no_long_in_batch_16": p_no_long, "expected_per_batch_long_position_fraction": expected_batch_long_share, "first_20_batch_long_counts": [b["long"] for b in small_batches], "note": "The algebra targets cumulative/global token totals; it does not equate per-update mean-loss coefficients to those totals."})

report_path = BASE / "inputs/docs/course-experiments/results/sft_ablation.json"
report = json.loads(report_path.read_bytes())
assert report["step_scale"] == 1.0 and report["seed"] == 42
for path in ["scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py"]:
    assert sha(ROOT / path) == report["code_sha256"][path] == sha(BASE / "inputs" / path)
tree = ast.parse((ROOT / "scripts/course_experiments/common.py").read_text())
fit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit_lm")
defaults = dict(zip([a.arg for a in fit.args.args][-len(fit.args.defaults):], fit.args.defaults))
assert ast.literal_eval(defaults["batch_size"]) == 16
records = {}
for name in ["attributes", "arithmetic"]:
    path = BASE / "inputs/original-training-records" / name / "train.jsonl"
    assert sha(path) == report["results"]["data"][name]["train"]["sha256"]
    records[name] = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records[name]) == report["results"]["data"][name]["train"]["records"]
    examples_for_source = text_examples(records[name], "sft", 128)
    show("original_source_lengths", {"source": name, "records": len(records[name]), "effective_label_count_histogram": dict(Counter(int((labels != IGNORE).sum()) for _, labels in examples_for_source)), "max_input_length": max(len(inputs) for inputs, _ in examples_for_source)})
for name, selected in [("b-only", records["arithmetic"]), ("replay", records["attributes"] + records["arithmetic"])]:
    recorded = report["results"]["runs"][name]
    assert records_sha256(selected) == recorded["training"]["records_sha256"]
    examples_for_run = text_examples(selected, "sft", 128)
    rng = random.Random(report["seed"])
    effective, draws, per_source = 0, 0, Counter()
    for _ in range(recorded["training"]["steps"]):
        # Same sampler primitive and order as fit_lm; counting only, no training.
        indices = rng.choices(range(len(examples_for_run)), k=16)
        batch = [examples_for_run[i] for i in indices]
        _, labels, _ = pad_batch(batch)
        effective += int((labels != IGNORE).sum())
        draws += len(batch)
        for i in indices:
            per_source["A" if name == "replay" and i < 45 else "B"] += 1
    assert effective == recorded["training"]["effective_tokens"]
    assert recorded["training"]["steps"] == 500 and draws == 8000
    show("original_empirical_counts", {"run": name, "steps": 500, "batch_size_from_original_contract": 16, "sampled_records": draws, "distinct_input_records": len(selected), "source_draw_counts": dict(per_source), "recounted_effective_labels": effective, "original_json_effective_labels": recorded["training"]["effective_tokens"], "matched_record_sha256": records_sha256(selected)})
show("original_empirical_ratio", {"replay_divided_by_b_only": 36384 / 18453, "increase_over_b_only": (36384 - 18453) / 18453, "original_revision": report["revision"], "original_device": report["device"], "current_check_device": "cpu", "training_rerun": False})
environment = {"python": sys.version, "python_executable": sys.executable, "torch": torch.__version__, "torch_git_version": torch.version.git_version, "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "device": "cpu", "threads": str(torch.get_num_threads()), "scope": "No model/optimizer/weights/data preparation; arithmetic, CPU loss tensors and original-record count-only sampler."}
(BASE / "cpu-environment.json").write_text(json.dumps(environment, indent=2) + "\n")
show("environment", environment)
print("ALL_ASSERTIONS_PASSED")
