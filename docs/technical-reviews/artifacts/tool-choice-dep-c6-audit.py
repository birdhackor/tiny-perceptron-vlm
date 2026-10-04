"""Bounded CPU audit of C.6 examples and the original finite-policy records.

Does not fit a model, load a checkpoint, call CUDA, or regenerate experiment data.
"""

import hashlib
import json
import platform
import re
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.course_experiments.applications import (
    _FinitePolicy,
    _POLICY_ACTIONS,
    _evaluate_policy,
    _policy_reward,
    _reasoning_records,
)
from scripts.course_experiments.common import records_sha256, split_records


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(text):
    clean = text.strip()
    return int(clean) if re.fullmatch(r"-?[0-9]+", clean) else None


torch.set_num_threads(1)
assert not torch.cuda.is_available()
original_path = "docs/course-experiments/results/reasoning.json"
original = json.loads(Path(original_path).read_text())
code_hashes = {}
for path in (
    "scripts/course_experiments/applications.py",
    "scripts/course_experiments/common.py",
):
    code_hashes[path] = digest(path)
    assert code_hashes[path] == original["code_sha256"][path]

examples = ["3", "4", "5", "答案是4"]
exercise = [
    {"text": text, "parsed": parse(text), "reward": float(parse(text) == 5)}
    for text in examples
]
assert [row["reward"] for row in exercise] == [0.0, 0.0, 1.0, 0.0]
boundary_cases = {
    " 4 ": 4,
    "\t-2\n": -2,
    "04": 4,
    "-0": 0,
    "+4": None,
    "４": None,
    "4.0": None,
    "4 4": None,
    "4\n5": None,
    "": None,
    "-": None,
}
assert {text: parse(text) for text in boundary_cases} == boundary_cases
assert _POLICY_ACTIONS == [str(i) for i in range(16)] + [" ".join(map(str, range(16)))]
finite_checks = []
for truth in range(16):
    strict = [_policy_reward(action, truth) for action in range(17)]
    proxy = [_policy_reward(action, truth, proxy=True) for action in range(17)]
    assert strict == [float(action == truth) for action in range(17)]
    assert proxy == [float(action == truth or action == 16) for action in range(17)]
    finite_checks.append({"truth": truth, "enumeration_strict": strict[16], "enumeration_proxy": proxy[16]})

splits = split_records(_reasoning_records(), original["seed"])
split_audit = {}
for name, rows in splits.items():
    split_audit[name] = {
        "records": len(rows),
        "families": len({row["family"] for row in rows}),
        "sha256": records_sha256(rows),
    }
    assert split_audit[name] == original["results"]["split"][name]
assert not ({row["family"] for row in splits["train"]} & {row["family"] for row in splits["test"]})

branch = original["results"]["reinforce"]["weak_proxy"]
training = branch["training"]
assert training["steps"] == 1200 and training["sampled_actions"] == 76800
assert training["effective_tokens"] == 0
traces = branch["after"]["samples"]
assert len(traces) == 24
assert {trace["question"] for trace in traces} == {row["question"] for row in splits["test"]}
counts = {"sample_accuracy": 0, "sample_proxy_reward": 0, "enumeration_action_rate": 0}
compact_original = []
for trace in traces:
    assert trace["truth"] == trace["a"] + trace["b"] + trace["c"]
    assert trace["family"] == ":".join(map(str, sorted((trace["a"], trace["b"], trace["c"]))))
    assert len(trace["samples"]) == 16
    assert len(trace["probabilities"]) == 17
    assert abs(sum(trace["probabilities"]) - 1.0) < 1e-6
    for sample in trace["samples"]:
        action = sample["action_id"]
        assert sample["generated"] == _POLICY_ACTIONS[action]
        strict = float(parse(sample["generated"]) == trace["truth"])
        proxy = float(str(trace["truth"]) in sample["generated"].split())
        assert strict == _policy_reward(action, trace["truth"])
        assert proxy == _policy_reward(action, trace["truth"], proxy=True)
        assert sample["strict_reward"] == strict and sample["proxy_reward"] == proxy
        counts["sample_accuracy"] += int(strict)
        counts["sample_proxy_reward"] += int(proxy)
        counts["enumeration_action_rate"] += int(action == 16)
    compact_original.append({
        "question": trace["question"],
        "truth": trace["truth"],
        "family": trace["family"],
        "action_ids": [sample["action_id"] for sample in trace["samples"]],
        "enumeration_probability": trace["probabilities"][16],
    })

summary = {}
for key, numerator in counts.items():
    summary[key] = {"numerator": numerator, "denominator": 384, "rate": numerator / 384}
    assert summary[key] == branch["after"][key]
assert summary["sample_accuracy"]["numerator"] == 0
assert summary["sample_proxy_reward"]["numerator"] == 384
assert summary["enumeration_action_rate"]["numerator"] == 384
example = next(trace for trace in traces if trace["question"] == "(0+3)+3=?")

# Exercise the actual CPU evaluator without training. This proves this local
# mapping/sampling behavior, not the old trained weights or arithmetic skill.
torch.manual_seed(42)
untrained_policy = _FinitePolicy().to("cpu")
probe = _evaluate_policy(untrained_policy, splits["test"][:2], SimpleNamespace(device="cpu"))
assert len(probe["samples"]) == 2
assert sum(len(trace["samples"]) for trace in probe["samples"]) == 32
for trace in probe["samples"]:
    for sample in trace["samples"]:
        assert sample["generated"] == _POLICY_ACTIONS[sample["action_id"]]

plan_path = "docs/course-experiments/plan.json"
plan = json.loads(Path(plan_path).read_text())
reasoning_entry = next(item for item in plan["sequence"] if item["id"] == "reasoning")
tool_entry = next(item for item in plan["supporting_experiments"] if item["id"] == "tool_choice")
assert "C.6" in reasoning_entry["lessons"]
assert reasoning_entry["evidence"] == original_path
assert tool_entry["lessons"] == ["B.5", "B.6", "B.7", "B.8"]

result = {
    "command": "CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/tool-choice-dep-c6-audit.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "original_report": {
        "path": original_path, "sha256": digest(original_path), "revision": original["revision"],
        "seed": original["seed"], "device": original["device"], "gpu": original["gpu"],
        "python": original["python_version"], "torch": original["torch_version"],
        "status": original["status"], "evidence_status": original["evidence_status"],
    },
    "original_code_hashes_match_current": code_hashes,
    "exercise_truth5": exercise,
    "parser_boundary_cases": [{"text": text, "parsed": value} for text, value in boundary_cases.items()],
    "predefined_actions": _POLICY_ACTIONS,
    "all_16_truths_checked": finite_checks,
    "split_regeneration_matches_original_hashes": split_audit,
    "train_test_families_disjoint": True,
    "weak_training_record": {key: training[key] for key in ("steps", "sampled_actions", "effective_tokens", "checkpoint")},
    "weak_after_recomputed_summary": summary,
    "original_24_test_traces_compact": compact_original,
    "original_example_literal": example,
    "fresh_untrained_cpu_evaluator_probe": probe,
    "current_plan": {"path": plan_path, "sha256": digest(plan_path), "reasoning": reasoning_entry, "tool_choice": tool_entry},
    "limits": [
        "The current audit verifies original stored sample records; it does not retrain or reload the original CUDA checkpoints.",
        "The 384 original sampled actions come from only 24 test questions, covering 6 unseen operand families at seed 42.",
        "The policy selects one of 17 predefined text actions; it generates no autoregressive tokens.",
        "The fresh evaluator probe uses random untrained CPU weights and supports only local sampling/mapping behavior.",
        "The parser checks a single bounded arithmetic answer and cannot establish explanation correctness or internal reasoning faithfulness.",
    ],
    "result": "All assertions passed; C.6 original code, exercise, all finite rewards, split hashes, 384 original samples and updated plan checked.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
