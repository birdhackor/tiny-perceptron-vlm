"""Bounded CPU arithmetic, loss-composition and original-result audit; no weights."""
import ast
import copy
import hashlib
import json
import math
import platform
import random
import sys
from fractions import Fraction
from pathlib import Path

import torch
from torch.nn import functional as F

OUT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
HIST = OUT / "inputs/historical"
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


ns = {"torch": torch, "F": F, "IGNORE": -100, "SPECIALS": ("<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<image>", "<audio>", "<system>"), "json": json, "random": random, "hashlib": hashlib}
load_definitions(HIST / "tiny_perceptron/data.py", ["ByteTokenizer", "render_chat"], ns)
load_definitions(HIST / "tiny_perceptron/model.py", ["loss_sum", "masked_loss"], ns)
load_definitions(HIST / "scripts/course_experiments/common.py", ["split_records"], ns)
load_definitions(HIST / "scripts/course_experiments/text.py", ["arithmetic_records"], ns)
load_definitions(HIST / "scripts/course_experiments/behavior.py", ["_conversation", "_style_record", "_style_metrics"], ns)

fence = (OUT / "code/original/fence-1.py").read_bytes()
original_ns = {}
exec(compile(fence, "course/chapters/08.md#8.7:fence-1", "exec"), original_ns)
original_answers = copy.deepcopy(original_ns["answers"])
weights = [0, 0.2, 0.49999, 0.5, 0.50001, 2]
scores = []
for w in weights:
    totals = [w * row["風格"] + row["正確"] for row in original_answers]
    exact = [Fraction(str(w)) * row["風格"] + row["正確"] for row in original_answers]
    assert all(abs(t - float(e)) < 1e-12 for t, e in zip(totals, exact))
    comparison = (totals[0] > totals[1]) - (totals[0] < totals[1])
    scores.append({"weight": w, "totals": totals, "rounded": [round(t, 1) for t in totals], "short_vs_wrong": comparison})
assert original_ns["answers"] == original_answers
assert [scores[i]["short_vs_wrong"] for i in [1, 2, 3, 4, 5]] == [1, 1, 0, -1, -1]
print("SCORING_BOUNDARIES", json.dumps(scores, ensure_ascii=False))

# Identical wrong arithmetic logit at position 0; add predictable target positions.
loss_checks = []
for length in (2, 44):
    logits = torch.tensor([0.0, 4.0], dtype=torch.float64).repeat(1, length, 1)
    labels = torch.tensor([[0] + [1] * (length - 1)])
    total, count = ns["loss_sum"](logits, labels)
    mean = ns["masked_loss"](logits, labels)
    per = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), reduction="none")
    expected = (math.log1p(math.exp(4)) + (length - 1) * math.log1p(math.exp(-4))) / length
    assert abs(float(mean) - expected) < 1e-12
    assert not bool((logits[0, 0].argmax() == labels[0, 0]).item())
    ignored_logits = torch.cat([logits, torch.tensor([[[100.0, -100.0]]], dtype=torch.float64)], dim=1)
    ignored_labels = torch.cat([labels, torch.tensor([[-100]])], dim=1)
    ignored_total, ignored_count = ns["loss_sum"](ignored_logits, ignored_labels)
    assert int(ignored_count) == length and float(ignored_total) == float(total)
    loss_checks.append({"valid_positions": int(count), "nll_sum_nats": float(total), "nll_mean_nats_per_valid_position": float(mean), "wrong_arithmetic_position_nll": float(per[0]), "arithmetic_argmax_correct": False, "ignore_minus_100_preserves_sum_count": True})
assert loss_checks[1]["nll_mean_nats_per_valid_position"] < loss_checks[0]["nll_mean_nats_per_valid_position"]
try:
    ns["loss_sum"](torch.zeros(1, 1, 2), torch.tensor([[-100]]))
except ValueError as error:
    empty_error = str(error)
else:
    raise AssertionError("zero denominator must raise ValueError")
print("LOSS_COMPOSITION", json.dumps(loss_checks, ensure_ascii=False))
print("ALL_IGNORED_BOUNDARY", empty_error)

record = json.loads((OUT / "inputs/current/docs/course-experiments/results/style.json").read_text())
parts = ns["split_records"](ns["arithmetic_records"](), seed=record["seed"])
assert {k: len(v) for k, v in parts.items()} == {"train": 49, "validation": 8, "test": 7}
families = {k: {r["family"] for r in v} for k, v in parts.items()}
assert all(not families[a] & families[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
empirical = {}
tok = ns["ByteTokenizer"]()
for style in ("concise", "vivid"):
    run = record["results"]["default_style_runs"][style]
    rows = [ns["_style_record"](r, style, False) for r in parts["test"]]
    # _save_splits hashes exact JSONL bytes, not _digest's canonical JSON bytes.
    jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    assert hashlib.sha256(jsonl).hexdigest() == run["data"]["test"]["sha256"]
    (OUT / f"results/reconstructed-{style}-test.jsonl").write_bytes(jsonl)
    test = run["after"]["test"]
    checked = copy.deepcopy(test)
    ns["_style_metrics"](checked, rows)
    assert checked["rubric"] == test["rubric"]
    details = []
    for row, sample in zip(rows, test["samples"], strict=True):
        assert row["messages"][:-1] == sample["messages"]
        assert row["messages"][-1]["content"] == sample["expected"]
        x, y = ns["render_chat"](row["messages"])
        count = int((y != -100).sum())
        expected_count = len(sample["expected"].encode("utf-8")) + 1
        assert count == expected_count and y[y != -100][-1] == tok.eos_id
        ids = sample["generated_ids"]
        assert ids[-1] == tok.eos_id and all(8 <= t < 264 for t in ids[:-1])
        assert tok.decode(ids[:-1]) == sample["generated"]
        assert tok.encode(sample["generated"]) == ids[:-1]
        generated_value = sample["generated"].split("，", 1)[0].strip()
        arithmetic_correct = generated_value == str(row["a"] + row["b"])
        assert arithmetic_correct == sample["content_correct"] == checked["samples"][len(details)]["content_correct"]
        details.append({"question": row["messages"][0]["content"], "expected": sample["expected"], "generated": sample["generated"], "target_positions_including_eos": count, "arithmetic_correct": arithmetic_correct, "style_correct": sample["style_correct"], "valid_raw_ids_and_eos": True})
    total_targets = sum(r["target_positions_including_eos"] for r in details)
    assert total_targets == test["effective_tokens"]
    assert abs(test["nll_sum"] / total_targets - test["nll"]) < 1e-14
    printed = 8.41357 if style == "concise" else 0.26805
    assert abs(test["nll"] - printed) <= 0.000005
    assert test["records"] == test["examples"] == len(rows) == len(details) == 7
    assert sum(r["arithmetic_correct"] for r in details) == 0
    assert sum(r["style_correct"] for r in details) == 7
    assert run["training"]["steps"] == 450 and run["training"]["records"] == 49
    empirical[style] = {"records": 7, "effective_target_positions": total_targets, "nll_sum": test["nll_sum"], "recalculated_nll": test["nll_sum"] / total_targets, "rounded_nll": round(test["nll"], 5), "content_correct": 0, "fixed_style_correct": 7, "details": details}
print("ORIGINAL_RESULT_AUDIT", json.dumps(empirical, ensure_ascii=False))

result = {
    "scope": "Original fence plus score-boundary and synthetic-logit CPU variants; recompute original JSON arithmetic/token denominators and historic rubric. No model inference, training, downloads, neural weights, or GPU rerun.",
    "environment": {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "cuda_build": str(torch.version.cuda), "device": "cpu", "platform": platform.platform(), "threads": str(torch.get_num_threads())},
    "scores": scores,
    "loss_composition": loss_checks,
    "all_ignored_boundary": empty_error,
    "original_empirical_result": empirical,
    "original_training_environment": {k: record[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status")},
    "historical_functions_executed": ["ByteTokenizer", "render_chat", "loss_sum", "masked_loss", "split_records", "arithmetic_records", "_conversation", "_style_record", "_style_metrics"],
    "checks": "all assertions passed",
}
(OUT / "results/verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("ALL_ASSERTIONS_PASSED")
