"""Fresh B.1 review: deterministic counts and preserved traces; no training."""

import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import random
import re
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from scripts.course_experiments.applications import _parse_json_action, _tool_records
from scripts.course_experiments.common import Context, new_lm, records_sha256, split_records, text_examples
from scripts.course_experiments.run import experiment_spec, list_assets
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.retrieval import call_tool

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def run_example(code):
    output = io.StringIO()
    scope = {}
    with contextlib.redirect_stdout(output):
        exec(compile(code, "B.1-original-fence", "exec"), scope)
    return {
        "stdout": output.getvalue(),
        "question": scope["training_example"]["user"],
        "request": scope["request"],
        "roundtrip": json.loads(scope["text"]) == scope["request"],
    }


raw = (ROOT / "course/chapters/0B.md").read_text()
body = re.search(r"^## B\.1 .*?(?=^## |\Z)", raw, re.M | re.S).group(0)
code = re.search(r"```python\n(.*?)```", body, re.S).group(1)
original = run_example(code)
mismatch = run_example(code.replace('"123乘45"', '"123乘46"'))
aligned = run_example(code.replace('"123乘45"', '"123乘46"').replace('"b": 45', '"b": 46'))
assert original["roundtrip"] and mismatch["roundtrip"] and aligned["roundtrip"]
assert mismatch["question"] == "123乘46" and mismatch["request"]["arguments"]["b"] == 45
assert aligned["question"] == "123乘46" and aligned["request"]["arguments"]["b"] == 46
calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call)]
assert calls == ["json.dumps", "print", "print", "print"]
arithmetic = {"123x45": call_tool(original["request"]),
              "123x46": call_tool(aligned["request"]),
              "123x54": call_tool({"name": "multiply", "arguments": {"a": 123, "b": 54}})}
assert arithmetic == {"123x45": 5535, "123x46": 5658, "123x54": 6642}

report = json.loads((ROOT / "docs/course-experiments/results/tools.json").read_text())
result = report["results"]
records = _tool_records()
splits = split_records(records, 42)
split_summary = {
    name: {"records": len(rows), "families": len({r["family"] for r in rows}),
           "sha256": records_sha256(rows)}
    for name, rows in splits.items()
}
assert split_summary == result["split"]
family_sets = [{r["family"] for r in rows} for rows in splits.values()]
assert all(not family_sets[i] & family_sets[j] for i in range(3) for j in range(i + 1, 3))
assert len(records) == 210 and len({r["family"] for r in records}) == 65
pair_families = {r["family"] for r in records if r["operation"] != "copy"}
copy_families = {r["family"] for r in records if r["operation"] == "copy"}
assert len(pair_families) == 55 and len(copy_families) == 10
assert not pair_families & copy_families
copy_three = next(r for r in records if r["family"] == "copy:3")
assert copy_three["messages"][1]["content"] == "COPY:3"
assert _parse_json_action(copy_three["messages"][2]["content"]) == {"done": True, "answer": 3}
assert "name" not in _parse_json_action(copy_three["messages"][2]["content"])
assert all(m["content"].isascii() for r in records for m in r["messages"])
assert max(ord(c) for r in records for m in r["messages"] for c in m["content"]) < 128

examples = text_examples(splits["train"], mode="sft", max_length=384)
tok = ByteTokenizer()
target_counts = [int((y != IGNORE).sum()) for _, y in examples]
for row, (_, y), count in zip(splits["train"], examples, target_counts):
    expected = sum(len(tok.encode(m["content"])) + 1 for m in row["messages"] if m["role"] == "assistant")
    assert count == expected
    assert int((y == tok.eos_id).sum()) == sum(m["role"] == "assistant" for m in row["messages"])
sampler = random.Random(42)
effective = sum(sum(sampler.choices(target_counts, k=16)) for _ in range(1200))
assert effective == result["training"]["effective_tokens"] == 1296064

ctx = Context("cpu", ROOT / "outputs", ROOT / "outputs", ROOT / "assets/training", seed=42)
model = new_lm(ctx, width=64, layers=2, max_length=384, heads=2, backend="sdpa")
assert model.description() == result["model"]
assert all(p.device.type == "cpu" for p in model.parameters())
same_seed_model = new_lm(ctx, width=64, layers=2, max_length=384, heads=2, backend="sdpa")
assert all(torch.equal(a, b) for a, b in zip(model.parameters(), same_seed_model.parameters()))
assert random.Random(42).sample(range(100), 20) == random.Random(42).sample(range(100), 20)

source_paths = ["scripts/course_experiments/applications.py", "scripts/course_experiments/common.py",
                "scripts/course_experiments/run.py", "tiny_perceptron/data.py",
                "tiny_perceptron/retrieval.py", "tiny_perceptron/training.py"]
hash_checks = {p: {"current": sha(p), "experiment": report["code_sha256"][p]} for p in source_paths}
assert all(v["current"] == v["experiment"] for v in hash_checks.values())
episodes = result["samples"]
assert len(episodes) == split_summary["test"]["records"]
trace_events = tool_calls = final_actions = 0
for episode in episodes:
    assert episode["question"] in {r["messages"][1]["content"] for r in splits["test"]}
    for event in episode["trace"]:
        if "generation" not in event:
            continue
        trace_events += 1
        generated = event["generation"]["samples"][0]["generated"]
        if "parsed" in event:
            action = _parse_json_action(generated)
            assert action == event["parsed"]
            if event.get("executed"):
                tool_calls += 1
                assert call_tool(action) == event["tool_result"]
            if event.get("done"):
                final_actions += 1
                assert action["answer"] == episode["answer"]
assert tool_calls == result["actual_tool_calls"]

tools_spec = experiment_spec("tools")
choice_spec = experiment_spec("tool_choice")
assert tools_spec["lessons"] == ["B.1", "B.2", "B.3", "B.4"]
assert choice_spec["lessons"] == ["B.5", "B.6", "B.7", "B.8"]
assert list_assets("tools") == []
artifact_names = sorted(a["path"] for a in report["artifacts"])
assert {"dataset.json", "generations.json", "model.pt", "model-step-300.pt", "model-step-600.pt",
        "model-step-900.pt", "model-step-1200.pt"} <= set(artifact_names)

prerequisites = {}
prerequisite_outputs = {}
for path, lesson in [("course/chapters/07.md", "7.1"), ("course/chapters/08.md", "8.5"),
                     ("course/chapters/07.md", "7.11")]:
    text = (ROOT / path).read_text()
    section = re.search(r"^## " + re.escape(lesson) + r" .*?(?=^## |\Z)", text, re.M | re.S).group(0)
    prerequisites[path + "#" + lesson] = hashlib.sha256(section.encode()).hexdigest()
    fence = re.search(r"```python\n(.*?)```", section, re.S).group(1)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(fence, path + "#" + lesson, "exec"), {})
    prerequisite_outputs[path + "#" + lesson] = output.getvalue()
assert "有效目標 [58, 2]" in prerequisite_outputs["course/chapters/07.md#7.1"]
assert "格式 True 內容 True" in prerequisite_outputs["course/chapters/08.md#8.5"]
assert "有效答案 4" in prerequisite_outputs["course/chapters/07.md#7.11"]

print(json.dumps({
    "reviewer_task": "/root/technical_dep_b1",
    "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "intro_sha256": hashlib.sha256(raw[:raw.index("## B.1 ")].encode()).hexdigest(),
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "original": original, "exercise_question_only": mismatch, "exercise_aligned": aligned,
    "original_call_names": calls, "arithmetic": arithmetic,
    "split_summary": split_summary, "family_overlap": False,
    "record_count": len(records), "family_count": 65,
    "family_derivation": {"unordered_operand_pairs_with_repetition": 10 * 11 // 2,
                          "pair_families": len(pair_families), "copy_families": len(copy_families),
                          "disjoint_pair_and_copy_names": True},
    "copy_three_target": copy_three["messages"][2]["content"],
    "same_seed_initialization_and_sampling_replayed": True,
    "all_message_contents_ascii": True,
    "sampled_training_targets": {"seed": 42, "batch_size": 16, "steps": 1200,
                                 "effective_targets": effective, "counts_include_assistant_eos": True,
                                 "maximum_record_input_length": max(len(x) for x, _ in examples)},
    "model_instantiation_without_training": model.description(), "code_hash_checks": hash_checks,
    "preserved_trace_audit": {"test_episodes": len(episodes), "generation_events": trace_events,
                              "tool_calls": tool_calls, "final_actions": final_actions,
                              "every_preserved_executed_result_recomputed": True},
    "formal_training_metadata": {"device": report["device"], "gpu": report["gpu"],
                                 "python": report["python_version"], "torch": report["torch_version"],
                                 "steps": result["training"]["steps"],
                                 "evidence_status": report["evidence_status"]},
    "tools_plan": tools_spec, "tool_choice_intro_navigation": choice_spec,
    "preserved_artifact_names": artifact_names, "prerequisite_sections": prerequisites,
    "prerequisite_example_outputs": prerequisite_outputs,
    "scope": "CPU review of current code, original example and exercises, deterministic sampling counts, and preserved L4 traces. No optimizer updates, checkpoint loading, formal rerun or GPU work."
}, ensure_ascii=False, indent=2))
