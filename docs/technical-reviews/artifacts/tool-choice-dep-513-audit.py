"""Fresh CPU evidence audit for 5.13; no training, downloads, or GPU work."""

import ast
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.course_experiments.run import experiment_spec
from tiny_perceptron.data import IGNORE, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM


def digest(data):
    return hashlib.sha256(data).hexdigest()


def functions(text):
    return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}


assert not torch.cuda.is_available() and torch.version.cuda is None
report_path = Path("docs/course-experiments/results/text_foundation.json")
report = json.loads(report_path.read_text())
plan_path = Path("docs/course-experiments/plan.json")
plan = json.loads(plan_path.read_text())
assert report["evidence_status"] == "complete_run" and report["status"] == "completed"
assert report["seed"] == 42 and report["step_scale"] == 1.0
assert "5.13" in report["results"]["sections"]
plan_entry = next(row for row in plan["sequence"] if row["id"] == "text_foundation")
assert plan_entry["module"] == "text" and plan_entry["function"] == "run_text_foundation"
assert plan_entry["evidence"] == report_path.as_posix() and plan_entry["status"] == "complete_run"
assert experiment_spec("text_foundation") == plan_entry

history = {}
current_functions = {}
for path, names in (
    ("scripts/course_experiments/text.py", ["_steps", "_save_splits", "_evaluations", "run_text_foundation"]),
    ("scripts/course_experiments/common.py", ["seed", "new_lm", "split_records", "records_sha256", "text_examples", "_nll", "fit_lm"]),
):
    original = subprocess.run(
        ["git", "show", report["revision"] + ":" + path], capture_output=True, text=True, check=True
    ).stdout
    current = Path(path).read_text()
    assert digest(original.encode()) == report["code_sha256"][path]
    old_nodes, new_nodes = functions(original), functions(current)
    same = {
        name: ast.dump(old_nodes[name], include_attributes=False) == ast.dump(new_nodes[name], include_attributes=False)
        for name in names
    }
    assert all(same.values())
    history[path] = {
        "recorded_sha256": report["code_sha256"][path],
        "historical_blob_sha256": digest(original.encode()),
        "current_sha256": digest(current.encode()),
        "function_ast_matches": same,
    }
    current_functions[path] = new_nodes
    if path.endswith("common.py"):
        old_evaluation, new_evaluation = old_nodes["evaluate_lm"], new_nodes["evaluate_lm"]
        # The full NLL computation precedes generation and is unchanged.
        old_before = ast.dump(ast.Module(body=old_evaluation.body[:4], type_ignores=[]), include_attributes=False)
        new_before = ast.dump(ast.Module(body=new_evaluation.body[:4], type_ignores=[]), include_attributes=False)
        assert old_before == new_before
        history[path]["evaluate_lm_nll_setup_matches"] = True
        history[path]["generation_change"] = "Byte cutoff changed to full Unicode character boundary; all scaling strings are ASCII, so prefixes are equal."

pool_node = next(
    node for node in current_functions["scripts/course_experiments/text.py"]["run_text_foundation"].body
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "pool" for target in node.targets)
)
pool = eval(compile(ast.Expression(pool_node.value), "run_text_foundation:pool", "eval"))
parts = split_records(pool, seed=42)
assert len(pool) == 80
family_sets = {split: {row["family"] for row in rows} for split, rows in parts.items()}
assert not family_sets["train"] & family_sets["validation"]
assert not family_sets["train"] & family_sets["test"]
assert not family_sets["validation"] & family_sets["test"]
splits = {}
for split, rows in parts.items():
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    observed = {
        "records": len(rows), "families": len(family_sets[split]), "sha256": digest(encoded),
        "raw_bytes": sum(len(row["text"].encode()) for row in rows),
        "effective_targets": sum(int((labels != IGNORE).sum()) for _, labels in text_examples(rows)),
        "family_order": [row["family"] for row in rows],
    }
    for key in ("records", "families", "sha256"):
        assert observed[key] == report["results"]["scaling_data"][split][key]
    splits[split] = observed
assert [splits[s]["records"] for s in ("train", "validation", "test")] == [64, 8, 8]

sampling = {}
for count in (16, 64):
    rows = parts["train"][:count]
    examples = text_examples(rows)
    sampler = random.Random(42)
    exposure = {row["family"]: 0 for row in rows}
    effective = 0
    lengths = []
    for _ in range(150):
        selected = sampler.choices(list(range(len(examples))), k=4)
        batch = [examples[index] for index in selected]
        _, labels, _ = pad_batch(batch)
        effective += int((labels != IGNORE).sum())
        for index in selected:
            exposure[rows[index]["family"]] += 1
            lengths.append(int((examples[index][1] != IGNORE).sum()))
    assert sum(exposure.values()) == 600
    assert effective == (20765 if count == 16 else 20623)
    sampling[str(count)] = {
        "records": count, "steps": 150, "batch_size": 4, "sampled_records": 600,
        "effective_tokens": effective, "mean_exposures_per_distinct_record": 600 / count,
        "full_training_mean_denominator": sum(int((y != IGNORE).sum()) for _, y in examples),
        "records_sha256": records_sha256(rows), "target_lengths_sampled": sorted(set(lengths)),
    }
    for width in (16, 32):
        result = report["results"]["scaling"][f"scaling-w{width}-n{count}"]
        assert result["training"]["effective_tokens"] == effective
        assert result["training"]["records_sha256"] == sampling[str(count)]["records_sha256"]
        assert result["training"]["steps"] == 150

parameters = {}
for width in (8, 16, 32):
    model = TinyLM(ModelConfig(width=width, layers=1))
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == model.description()["parameters"] == 12 * width * width + 667 * width
    parameters[str(width)] = {"parameters": count, "config": model.description()["config"]}
assert [parameters[str(w)]["parameters"] for w in (8, 16, 32)] == [6104, 13744, 33632]

text = Path("outputs/tool-choice-dep-513/section.md").read_text()
table = re.findall(r"^\| (16|32) \| (16|64)篇 \| ([\d,]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$", text, re.M)
assert len(table) == 4
table_checks = {}
for width, count, effective, train, validation, test in table:
    key = f"scaling-w{width}-n{count}"
    row = report["results"]["scaling"][key]
    assert row["parameters"] == parameters[width]["parameters"]
    assert row["training"]["effective_tokens"] == int(effective.replace(",", ""))
    observed = [row["training"]["final_loss"], row["evaluation"]["validation"]["nll"], row["evaluation"]["test"]["nll"]]
    displayed = [float(train), float(validation), float(test)]
    assert all(abs(a-b) < 0.000005 for a,b in zip(observed, displayed))
    recomputed = {}
    for split in ("validation", "test"):
        metric = row["evaluation"][split]
        assert metric["records"] == 8 and metric["effective_tokens"] == splits[split]["effective_targets"]
        assert abs(metric["nll"] - metric["nll_sum"] / metric["effective_tokens"]) < 1e-14
        recomputed[split] = {k: metric[k] for k in ("nll", "nll_sum", "effective_tokens", "records")}
        for record in parts[split]:
            assert record["text"].isascii()
            assert record["text"][:24].encode() == record["text"].encode()[:24]
    table_checks[key] = {"displayed_losses": displayed, "recorded_losses": observed, "rounding_tolerance": 0.000005, "evaluation": recomputed}
for width in (16, 32):
    small = report["results"]["scaling"][f"scaling-w{width}-n16"]
    large = report["results"]["scaling"][f"scaling-w{width}-n64"]
    assert large["evaluation"]["validation"]["nll"] < small["evaluation"]["validation"]["nll"]
    assert large["training"]["final_loss"] > small["training"]["final_loss"]

original_code = Path("outputs/tool-choice-dep-513/fence-1.py").read_text()
exercise_outputs = {}
for steps in (200, 100):
    changed = original_code.replace(
        "batch_size, answers_per_record, steps = 4, 8, 100",
        "batch_size, answers_per_record, steps = 4, 8, 100\n"
        "        if width == 8 and records == 16:\n"
        f"            answers_per_record, steps = 4, {steps}",
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(changed, "5.13:exercise", "exec"), {})
    lines = buffer.getvalue().splitlines()
    values = [ast.literal_eval(line) for line in lines]
    assert values[0]["有效答案預算"] == (3200 if steps == 200 else 1600)
    assert all(value["有效答案預算"] == 3200 for value in values[1:])
    exercise_outputs[str(steps)] = values

print(json.dumps({
    "reviewer_task": "/root/technical_dep_513",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "cuda_available": False},
    "command_scope": "Existing report arithmetic and source audit, model construction, deterministic sampling and exercise only; no optimization or GPU execution.",
    "section_sha256": digest(Path("outputs/tool-choice-dep-513/section.md").read_bytes()),
    "current_plan_sha256": digest(plan_path.read_bytes()), "current_plan_entry": plan_entry,
    "report_sha256": digest(report_path.read_bytes()),
    "recorded_run": {k: report[k] for k in ("revision", "seed", "device", "gpu", "python_version", "torch_version", "step_scale", "evidence_status")},
    "historical_code_comparison": history, "splits": splits, "sampling": sampling, "parameters": parameters,
    "table_checks": table_checks,
    "toy_budget": {"original": 4 * 8 * 100, "exercise_steps": 3200 / (4 * 4), "unchanged_steps": 4 * 4 * 100},
    "linear_flops": {"width8": 2 * 8 * 8, "width16": 2 * 16 * 16, "ratio": (2 * 16 * 16)/(2 * 8 * 8)},
    "exercise_outputs": exercise_outputs,
    "all_assertions_passed": True,
}, ensure_ascii=False, indent=2))
