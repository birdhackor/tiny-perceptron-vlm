"""Independent B.7 audit: recompute saved facts and infer from a frozen checkpoint.

Run from the repository root; this script does not train or mutate a checkpoint.
"""
import hashlib
import io
import json
import platform
import random
import re
import time
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.course_experiments import tool_choice
from scripts.course_experiments.applications import _sample
from scripts.course_experiments.common import Context, records_sha256, text_examples
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat
from tiny_perceptron.training import load_checkpoint

ROOT = Path.cwd()
ART = ROOT / "docs/technical-reviews/artifacts"
NATIVE = ROOT / "outputs/tool-choice-development/fixed-900"
OUTPUT = ART / "tool-choice-b7-audit.json"
COMMAND = "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/tool-choice-b7-audit.py"
torch.set_num_threads(2)
tok = ByteTokenizer()
out = {
    "reviewer_task": "/root/technical_tool_b7",
    "command": COMMAND,
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "cpu_threads": "2"},
    "retraining": False,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_key(row):
    return tuple((m["role"], m["content"]) for m in row["messages"][:-1])


def decode(sample):
    ids = sample["generated_ids"]
    ended = bool(ids and ids[-1] == 2)
    body = ids[:-1] if ended else ids
    invalid = [i for i in body if i < 8]
    text = bytes(i - 8 for i in body if i >= 8).decode("utf-8", errors="replace")
    assert text == sample["generated"]
    assert ended == sample["eos"] and invalid == sample["invalid_special_tokens"]
    return text if ended and not invalid and text in ("DIRECT", "TOOL", "ASK") else None


def fraction(n, d):
    return {"numerator": n, "denominator": d, "rate": n / d if d else None}


def recount(rows, actions):
    # Independent implementation: do not invoke the experiment's summarize().
    assert len(rows) == len(actions)
    expected = [r["expected_action"] for r in rows]
    needed = [i for i, a in enumerate(expected) if a == "TOOL"]
    other = [i for i, a in enumerate(expected) if a != "TOOL"]
    off = [i for i, r in enumerate(rows) if not r["calculator_available"]]
    return {
        "accuracy": fraction(sum(a == b for a, b in zip(expected, actions, strict=True)), len(rows)),
        "valid_action": fraction(sum(a in ("DIRECT", "TOOL", "ASK") for a in actions), len(rows)),
        "needed_tool_selected": fraction(sum(actions[i] == "TOOL" for i in needed), len(needed)),
        "needed_tool_missed": fraction(sum(actions[i] != "TOOL" for i in needed), len(needed)),
        "unnecessary_tool_selected": fraction(sum(actions[i] == "TOOL" for i in other), len(other)),
        "unavailable_tool_selected": fraction(sum(actions[i] == "TOOL" for i in off), len(off)),
        "confusion": {a: dict(Counter(actions[i] or "INVALID" for i, e in enumerate(expected) if e == a)) for a in ("DIRECT", "TOOL", "ASK")},
        "by_intent": {intent: fraction(sum(e == a for r, e, a in zip(rows, expected, actions, strict=True) if r["intent"] == intent), sum(r["intent"] == intent for r in rows)) for intent in sorted({r["intent"] for r in rows})},
        "by_availability": {str(available).lower(): fraction(sum(e == a for r, e, a in zip(rows, expected, actions, strict=True) if r["calculator_available"] == available), sum(r["calculator_available"] == available for r in rows)) for available in (True, False)},
    }


source = (ROOT / "course/chapters/0B.md").read_text()
match = re.search(r"^## B\.7 .+?(?=^## |\Z)", source, re.M | re.S)
body = match[0]
out["section_sha256"] = hashlib.sha256(body.encode()).hexdigest()
out["input_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in [
    ROOT / "scripts/course_experiments/tool_choice.py",
    ROOT / "scripts/course_experiments/common.py",
    ROOT / "scripts/course_experiments/applications.py",
    ROOT / "tiny_perceptron/data.py",
    ROOT / "tiny_perceptron/model.py",
    ROOT / "tiny_perceptron/attention.py",
    ROOT / "tiny_perceptron/modern.py",
    ROOT / "tiny_perceptron/training.py",
    ROOT / "docs/course-experiments/results/tool_choice.json",
    NATIVE / "dataset.json", NATIVE / "results.json", NATIVE / "tool-choice.pt",
]}
public = json.loads((ROOT / "docs/course-experiments/results/tool_choice.json").read_text())
native = json.loads((NATIVE / "results.json").read_text())
assert public["results"] == native
assert public["code_sha256"]["scripts/course_experiments/tool_choice.py"] == digest(ROOT / "scripts/course_experiments/tool_choice.py")
assert public["results"]["reproducibility"]["script_sha256"] == digest(ROOT / "scripts/course_experiments/tool_choice.py")
out["public_equals_native_results"] = True
dataset = json.loads((NATIVE / "dataset.json").read_text())
rebuilt = tool_choice.split_records(tool_choice.build_records(), seed_value=42)
test_families = {r["family"] for r in rebuilt["test"]}
rebuilt["paraphrase_diagnostic"] = [r for r in tool_choice.build_records(paraphrase=True) if r["family"] in test_families]
assert dataset == rebuilt
out["dataset_exactly_rebuilt_with_current_source_and_seed42"] = True
split_info = {}
for name, rows in dataset.items():
    families = sorted({r["family"] for r in rows})
    by_family = {}
    for family in families:
        selected = [r for r in rows if r["family"] == family]
        assert len(selected) == (8 if name == "paraphrase_diagnostic" else 16)
        tasks = Counter(r["intent"] for r in selected)
        assert set(tasks) == {"numerical_addition", "explanation", "copy", "missing_quantity"}
        assert set(tasks.values()) == {2 if name == "paraphrase_diagnostic" else 4}
        for template in sorted({r["template"] for r in selected}):
            paired = [r for r in selected if r["template"] == template]
            assert len(paired) == 2 and {r["calculator_available"] for r in paired} == {True, False}
            assert paired[0]["messages"][1] == paired[1]["messages"][1]
        by_family[family] = {"records": len(selected), "intent_counts": dict(tasks), "template_ids": sorted({r["template"] for r in selected})}
    for row in rows:
        intent = row["intent"]
        want = ("TOOL" if row["calculator_available"] else "ASK") if intent == "numerical_addition" else "ASK" if intent == "missing_quantity" else "DIRECT"
        assert row["expected_action"] == row["messages"][-1]["content"] == want
        assert row["messages"][0]["content"] == ("計算器可用。" if row["calculator_available"] else "計算器停用。")
        assert row["messages"][1]["role"] == "user" and not any(h in row["messages"][1]["content"] for h in ("CALC", "COPY"))
    checksum = records_sha256(rows)
    assert checksum == native["data"][name]["sha256"]
    split_info[name] = {"records": len(rows), "family_count": len(families), "families": families, "records_sha256": checksum, "action_counts": dict(Counter(r["expected_action"] for r in rows)), "by_family": by_family}
out["splits"] = split_info
for i, a in enumerate(("train", "validation", "test")):
    for b in ("train", "validation", "test")[i + 1:]:
        assert not ({r["family"] for r in dataset[a]} & {r["family"] for r in dataset[b]})
        assert not ({row_key(r) for r in dataset[a]} & {row_key(r) for r in dataset[b]})
assert "pair:1:2" in test_families and len(test_families) == 6
assert {r["family"] for r in dataset["paraphrase_diagnostic"]} == test_families
assert not ({row_key(r) for r in dataset["paraphrase_diagnostic"]} & {row_key(r) for n in ("train", "validation", "test") for r in dataset[n]})
out["split_identity_checks"] = {"train_validation_test_families_disjoint": True, "same_system_user_prompt_disjoint_across_splits": True, "diagnostic_uses_test_families_with_distinct_prompts": True, "train_operand_values": sorted({v for r in dataset["train"] for v in r["operands"]}), "unordered_families": sum(10 - a for a in range(10)), "diagonal_families": sum(1 for a in range(10))}
assert out["split_identity_checks"]["train_operand_values"] == list(range(10))

code = re.search(r"```python\n(.*?)\n```", body, re.S)[1]
scope = {}
buf = io.StringIO()
with redirect_stdout(buf):
    exec(compile(code, "B.7-literal-code", "exec"), scope)
assert (scope["correct"], scope["missed"], scope["extra"], scope["needed"], scope["not_needed"]) == (3, 1, 1, 2, 4)
original = buf.getvalue()
exercise = code.replace('predicted = ["TOOL", "DIRECT", "TOOL", "DIRECT", "ASK", "DIRECT"]', 'predicted = ["TOOL", "TOOL", "TOOL", "DIRECT", "ASK", "DIRECT"]')
buf = io.StringIO()
with redirect_stdout(buf):
    exec(compile(exercise, "B.7-exercise", "exec"), scope)
assert (scope["correct"], scope["missed"], scope["extra"], scope["needed"], scope["not_needed"]) == (4, 0, 1, 2, 4)
try:
    list(zip([1, 2], [3], strict=True))
except ValueError as exc:
    mismatch_error = str(exc)
else:
    raise AssertionError("Unequal lengths did not raise ValueError")
out["literal_and_exercise"] = {"original_stdout": original, "exercise_stdout": buf.getvalue(), "unequal_lengths_error": mismatch_error, "boolean_sum": sum([True, False, True]), "exercise_remaining_errors": [{"index": i, "gold": a, "predicted": b} for i, (a, b) in enumerate(zip(scope["gold"], scope["predicted"], strict=True)) if a != b]}

model, checkpoint = load_checkpoint(NATIVE / "tool-choice.pt", "cpu")
assert checkpoint["step"] == 900 and checkpoint["metadata"]["seed"] == 42
assert checkpoint["metadata"]["schedule"] == "constant" and checkpoint["metadata"]["lr"] == 0.003
assert checkpoint["training_state"]["batch_size"] == 24 and checkpoint["training_state"]["planned_steps"] == 900
assert checkpoint["config"] == native["model"]["config"]
assert digest(NATIVE / "tool-choice.pt") == native["frozen_evaluation"]["checkpoint_sha256"]
params = {name: {"shape": list(p.shape), "count": p.numel(), "trainable": p.requires_grad} for name, p in model.named_parameters()}
parameters = sum(p["count"] for p in params.values())
formula = 2 * 264 * 64 + 160 * 64 + 2 * (4 * 64 * 64 + 2 * 2 * 64 + (64 * 256 + 256) + (256 * 64 + 64)) + 2 * 64
assert formula == parameters == 143616 == native["training"]["parameters"]
assert all(p["trainable"] for p in params.values())
optimizer_steps = sorted({int(v["step"].item()) for v in checkpoint["optimizer"]["state"].values()})
assert optimizer_steps == [900]
assert {g["lr"] for g in checkpoint["optimizer"]["param_groups"]} == {0.003}
out["checkpoint"] = {"step": checkpoint["step"], "metadata": checkpoint["metadata"], "training_state_without_large_rng": {k: v for k, v in checkpoint["training_state"].items() if k != "sampler_rng"}, "config": checkpoint["config"], "optimizer_steps": optimizer_steps, "parameters": parameters, "parameter_formula": "2*264*64+160*64+2*(4*64*64+2*2*64+(64*256+256)+(256*64+64))+2*64=143616", "named_parameters": params}

examples = text_examples(dataset["train"], mode="sft", max_length=160)
target_lengths = [int((y != IGNORE).sum()) for x, y in examples]
for row, (_, y), n in zip(dataset["train"], examples, target_lengths, strict=True):
    valid_ids = y[y != IGNORE].tolist()
    assert valid_ids == tok.encode(row["expected_action"]) + [tok.eos_id]
    assert n == len(row["expected_action"]) + 1
sampler = random.Random(42)
total = 0
draw_counts = Counter()
milestones = []
for step in range(1, 901):
    chosen = sampler.choices(list(range(len(examples))), k=24)
    total += sum(target_lengths[i] for i in chosen)
    draw_counts.update(dataset["train"][i]["expected_action"] for i in chosen)
    if step in (225, 450, 675, 900):
        path = NATIVE / f"tool-choice-step-{step}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["step"] == step and payload["metadata"]["effective_tokens"] == total
        assert payload["training_state"]["sampler_rng"] == sampler.getstate()
        milestones.append({"step": step, "sha256": digest(path), "effective_tokens_recounted": total, "sampler_rng_matches": True})
assert total == 121297 == checkpoint["metadata"]["effective_tokens"] == native["training"]["effective_tokens"]
assert checkpoint["training_state"]["sampler_rng"] == sampler.getstate()
assert sum(draw_counts.values()) == 900 * 24 == 21600
assert total == sum(draw_counts[a] * (len(a) + 1) for a in draw_counts)
out["target_replay"] = {"effective_answer_positions": total, "sample_draws_with_replacement": 21600, "sampled_action_counts": dict(draw_counts), "target_positions_per_action": {a: len(a) + 1 for a in ("DIRECT", "TOOL", "ASK")}, "maximum_train_input_length": max(len(x) for x, y in examples), "final_sampler_rng_matches": True, "milestones": milestones}
_, b6labels = render_chat([{"role": "system", "content": "精確計算用工具，解釋或照抄直接回答；缺資訊或工具先求助。計算器可用。"}, {"role": "user", "content": "1加2等於多少？"}, {"role": "assistant", "content": "TOOL"}])
assert int((b6labels != IGNORE).sum()) == 5
out["b6_target_unit"] = {"effective_positions": 5, "ids": b6labels[b6labels != IGNORE].tolist()}

frozen_before = tool_choice._state_sha256(model)
assert frozen_before == native["frozen_evaluation"]["weights_sha256"]
ctx = Context("cpu", NATIVE, NATIVE, ROOT / "assets", 42)
evaluations = {}
started = time.perf_counter()
for name in ("validation", "test", "paraphrase_diagnostic"):
    saved = native[name]
    assert [s["record"] for s in saved["samples"]] == dataset[name]
    saved_actions = [decode(s["generation"]["samples"][0]) for s in saved["samples"]]
    metrics = recount(dataset[name], saved_actions)
    assert metrics == saved["metrics"]
    baselines = {"always_direct": recount(dataset[name], ["DIRECT"] * len(dataset[name])), "always_tool": recount(dataset[name], ["TOOL"] * len(dataset[name]))}
    assert baselines == saved["baselines"]
    regenerated = []
    for row, sample in zip(dataset[name], saved["samples"], strict=True):
        generation = _sample(model, row["messages"][:-1], ctx, tokens=8)
        s = generation["samples"][0]
        assert s["generated_ids"] == sample["generation"]["samples"][0]["generated_ids"]
        assert generation["input_ids"] == sample["generation"]["input_ids"]
        regenerated.append({"family": row["family"], "template": row["template"], "intent": row["intent"], "calculator_available": row["calculator_available"], "question": row["messages"][1]["content"], "expected_action": row["expected_action"], "chosen_action": decode(s), "generated_ids": s["generated_ids"], "eos": s["eos"], "invalid_special_tokens": s["invalid_special_tokens"]})
    live_actions = [r["chosen_action"] for r in regenerated]
    assert live_actions == saved_actions and recount(dataset[name], live_actions) == metrics
    evaluations[name] = {"metrics_independently_recomputed": metrics, "baselines_independently_recomputed": baselines, "fresh_inference_matches_saved_ids": len(regenerated), "per_record": regenerated}
out["evaluations"] = evaluations
out["frozen_inference_seconds"] = time.perf_counter() - started
after = tool_choice._state_sha256(model)
assert after == frozen_before
out["weights_sha256_before_and_after"] = {"before": frozen_before, "after": after, "unchanged": True}
train_seconds = native["training"]["seconds"]
total_seconds = native["total_seconds"]
assert round(train_seconds, 2) == 35.55 and round(total_seconds, 2) == 39.41
assert public["elapsed_seconds"] == total_seconds
out["recorded_original_timing"] = {"training_seconds": train_seconds, "whole_run_seconds": total_seconds, "rounded_training_seconds": round(train_seconds, 2), "rounded_whole_run_seconds": round(total_seconds, 2), "timing_scope_from_public_report": public["timing_scope"], "training_scope_from_source": "common.fit_lm starts immediately before update loop; includes optimizer updates and intermediate saves, excludes initial/final full-train NLL, final checkpoint save, and held-out inference", "whole_run_scope_from_source": "tool_choice.run starts before seed/data/model creation; ends after training, validation/test/diagnostic inference and final hash checks, before results.json serialization/packaging", "historical_timing_not_remeasured_by_this_audit": True}
out["scope_checks"] = {"assistant_training_targets_only_action_labels": True, "ask_reasons_are_record_metadata_not_model_outputs": True, "system_contains_status_only": True, "all_diagnostic_needed_tool_predictions": [r for r in evaluations["paraphrase_diagnostic"]["per_record"] if r["expected_action"] == "TOOL"], "no_new_tool_arguments_or_final_answers_evaluated": True, "does_not_establish_confidence_estimation_or_general_language_ability": True, "checkpoint_selection_scope": "For this preserved run, fixed PLANNED_STEPS=900 is passed once to fit_lm, evaluations follow final training, and final checkpoint/weights hashes match. This does not independently reconstruct undocumented activity outside that run."}
out["result"] = "All independent assertions passed; literal/exercise, data families, metrics/baselines, all 224 saved raw generations, 900-step target replay, checkpoints, parameters, frozen CPU inference and original timing scope agree. No training performed."
OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: out[k] for k in ("result", "section_sha256", "target_replay", "recorded_original_timing", "weights_sha256_before_and_after")}, ensure_ascii=False, indent=2))
