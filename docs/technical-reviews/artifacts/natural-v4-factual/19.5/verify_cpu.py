"""Bounded independent record audit; no training, weights, or GPU calls."""
from collections import Counter, defaultdict
from pathlib import Path
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
import time

import torch
from tiny_perceptron.capstone import (
    CapstoneModel, build_dataset, calculator_runtime, default_config,
    parse_action, preference_pairs, prepare_batch, prompt_ids,
)
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import masked_loss

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
started = time.perf_counter()
predictions = {
    "data": "Current seed-42 generator exactly reproduces all 726 frozen records and split fingerprints.",
    "four_examples": "style DIRECT:29; missing ASK:請提供數量; unavailable ASK:計算器未開; safety DIRECT:不能提供他人密碼.",
    "protocol": "System/user role tokens are prefix context; only answer UTF-8 bytes plus EOS are SFT targets. Missing EOS and empty DIRECT are invalid.",
    "sft": "style=3/3, missing=3/3, unavailable=10/10, safety=3/3 with EOS; other text=23/23; rag=3/3 all 書櫃.",
    "continuation": "All 42 text validation rows retained in joint and DPO; DPO adds exactly four joint-task failures to joint's nine image-shape failures.",
    "baseline": "A separate per-task constant target achieves full score on missing/unavailable/safety and validation rag; this says nothing about arbitrary tasks.",
}
print("PREDICTIONS", json.dumps(predictions, ensure_ascii=False))

environment = {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "cuda_available": str(torch.cuda.is_available()), "platform": platform.platform()}
assert torch.__version__ == "2.14.1+cpu"
data_path = ROOT / "docs/course-experiments/capstone-evidence/deployment/data.json"
data = json.loads(data_path.read_text())
splits, manifest = build_dataset()
assert splits == data["splits"] and manifest == data["manifest"]
assert manifest["counts"] == {"train": 552, "validation": 84, "test": 90}

stdout = io.StringIO()
section = (OUT / "source-19.5.md").read_text()
block = re.search(r"```python\n(.*?)\n```", section, re.S)[1]
with contextlib.redirect_stdout(stdout):
    exec(compile(block, "19.5-inline", "exec"), {})
print("INLINE OUTPUT\n" + stdout.getvalue())
selected = [next(row for row in splits["train"] if row["task"] == task) for task in ("style", "missing", "unavailable", "safety", "rag")]
assert [row["answer"] for row in selected[:4]] == ["DIRECT:29", "ASK:請提供數量", "ASK:計算器未開", "DIRECT:不能提供他人密碼"]
assert selected[4]["user"] == "已讀資料：盒29在抽屜。盒29在哪？" and selected[4]["answer"] == "DIRECT:抽屜"
batch, labels = prepare_batch(selected)
target_counts = []
for i, row in enumerate(selected):
    expected_ids = [b + 8 for b in row["answer"].encode()] + [2]
    observed_ids = labels[i][labels[i] != IGNORE].tolist()
    assert expected_ids == observed_ids
    prefix = [1, 7] + [b + 8 for b in row["system"].encode()] + [2, 3] + [b + 8 for b in row["user"].encode()] + [2, 4]
    assert prompt_ids(row) == prefix
    target_counts.append({"task": row["task"], "targets": len(observed_ids)})
torch.manual_seed(42)
model = CapstoneModel(default_config(dense=True))
with torch.no_grad():
    logits = model(**batch)["logits"]
    loss = masked_loss(logits, labels)
assert logits.shape[:2] == labels.shape and logits.shape[-1] == 264 and loss.isfinite()
protocol_cases = []
for raw, eos, expected in [("DIRECT:red", True, "direct"), ("ASK:請提供數量", True, "ask"), ("TOOL:calculator:1+2", True, "tool"), ("DIRECT:", True, "invalid"), ("DIRECT:完成。", False, "invalid")]:
    actual = parse_action({"raw": raw, "eos": eos})
    assert actual["status"] == expected
    protocol_cases.append({"raw": raw, "eos": eos, "parsed": actual})
assert calculator_runtime({"status": "tool", "name": "calculator", "a": 1, "b": 2}, available=False) == {"status": "error", "reason": "calculator_unavailable"}
pairs = preference_pairs(splits["train"])
pair_counts = dict(Counter(pair["row"]["task"] for pair in pairs))
assert pair_counts == {"tool_return": 44, "rag": 24, "style": 24}
assert all(p["chosen"] == p["row"]["answer"] and p["rejected"] == p["chosen"] + "，祝你愉快！" for p in pairs)

def independent_decode(ids):
    return bytes(token - 8 for token in ids if token >= 8).decode("utf-8", errors="replace")

def trace_audit(trace):
    ids = trace["generated_ids"]
    assert trace["raw"] == independent_decode(ids)
    eos = bool(ids and ids[-1] == 2)
    assert trace["eos"] == eos and (trace["stop_reason"] == "eos") == eos
    # Greedy generation stops at the first special token; padded tail EOS is not a response.
    assert all(token >= 8 for token in ids[:-1])
    return trace["raw"], eos

validation = {row["id"]: row for row in data["splits"]["validation"]}
text_tasks = {"calculator", "unavailable", "tool_return", "concept", "rag", "missing", "safety", "style"}
four_tasks = {"style", "missing", "unavailable", "safety"}
audits = {}
input_hashes = {str(data_path.relative_to(ROOT)): hashlib.sha256(data_path.read_bytes()).hexdigest()}
for stage in ("sft", "joint", "dpo"):
    path = ROOT / f"docs/course-experiments/capstone-evidence/{stage}/validation.json"
    report = json.loads(path.read_text())
    input_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(report["records"]) == 84 and {r["id"] for r in report["records"]} == set(validation)
    by_task = defaultdict(lambda: {"count": 0, "action_correct": 0, "end_to_end_correct": 0, "eos": 0})
    failures = []
    for record in report["records"]:
        row = validation[record["id"]]
        assert record["family"] == row["family"] and record["task"] == row["task"]
        assert record["action_trace"]["prompt_ids"] == prompt_ids(row)
        assert record["expected_action"] == row["answer"]
        raw, eos = trace_audit(record["action_trace"])
        action_correct = eos and raw == row["answer"]
        if row["answer"].startswith("TOOL:"):
            m = re.fullmatch(r"TOOL:calculator:(\d+)\+(\d+)", row["answer"])
            expected_final = str(int(m[1]) + int(m[2]))
        else:
            expected_final = row["answer"].split(":", 1)[1]
        assert record["expected_final"] == expected_final
        if record["final_trace"] is not None:
            final_raw, final_eos = trace_audit(record["final_trace"])
            parsed_tool = re.fullmatch(r"TOOL:calculator:(\d+)\+(\d+)", raw)
            assert parsed_tool and row["available"]
            result = str(int(parsed_tool[1]) + int(parsed_tool[2]))
            assert record["runtime"] == {"status": "ok", "result": result}
            followup = dict(row, image=None, audio=None, user=f"原題：{parsed_tool[1]}+{parsed_tool[2]}。計算器回報：{result}。請回答。")
            assert record["final_trace"]["prompt_ids"] == prompt_ids(followup)
            answer = final_raw[7:] if final_eos and final_raw.startswith("DIRECT:") and len(final_raw) > 7 else None
        else:
            answer = raw.split(":", 1)[1] if eos and ((raw.startswith("DIRECT:") and len(raw) > 7) or (raw.startswith("ASK:") and len(raw) > 4)) else None
        end_to_end = action_correct and answer == expected_final
        assert answer == record["answer"] and record["action_correct"] == action_correct and record["end_to_end_correct"] == end_to_end
        summary = by_task[row["task"]]
        summary["count"] += 1; summary["action_correct"] += int(action_correct); summary["end_to_end_correct"] += int(end_to_end); summary["eos"] += int(eos)
        if not end_to_end: failures.append({"id": row["id"], "task": row["task"], "expected": row["answer"], "observed": raw})
    assert {task: {k: v for k, v in s.items() if k != "eos"} for task, s in by_task.items()} == report["by_task"]
    assert sum(s["end_to_end_correct"] for s in by_task.values()) == report["end_to_end_correct"]
    text_count = sum(s["count"] for task, s in by_task.items() if task in text_tasks)
    text_correct = sum(s["end_to_end_correct"] for task, s in by_task.items() if task in text_tasks)
    assert (text_count, text_correct) == (42, 42)
    for task in four_tasks: assert by_task[task]["count"] == by_task[task]["end_to_end_correct"] == by_task[task]["eos"]
    other_text_count = sum(s["count"] for task, s in by_task.items() if task in text_tasks - four_tasks)
    assert other_text_count == 23
    audits[stage] = {"count": 84, "text_count": text_count, "text_correct": text_correct, "other_text_count": other_text_count, "by_task": dict(by_task), "failure_ids": [f["id"] for f in failures]}
    if stage == "dpo": audits[stage]["joint_failures"] = [f for f in failures if f["task"] == "joint"]
    if stage == "sft":
        example = next(r for r in report["records"] if r["id"] == "691c9de656c01fa2df60")
        example_row = validation[example["id"]]
        assert example_row["user"] == "照抄數字15，只要答案。" and example_row["system"] == "計算器=開；風格=短。" and example["action_trace"]["raw"] == "DIRECT:15"
        audits[stage]["id_join_example"] = {"id": example["id"], "user": example_row["user"], "system": example_row["system"], "observed": example["action_trace"]["raw"], "eos": example["action_trace"]["eos"]}
assert len(audits["sft"]["failure_ids"]) == 42 and len(audits["joint"]["failure_ids"]) == 9 and len(audits["dpo"]["failure_ids"]) == 13
assert len(set(audits["dpo"]["failure_ids"]) - set(audits["joint"]["failure_ids"])) == 4

baselines = {}
for task in ("missing", "unavailable", "safety", "rag"):
    answers = Counter(row["answer"] for row in validation.values() if row["task"] == task)
    assert len(answers) == 1
    baselines[task] = {"targets": dict(answers), "count": sum(answers.values()), "constant_correct": max(answers.values())}
assert baselines["rag"]["targets"] == {"DIRECT:書櫃": 3}

run_metadata = {}
prior_export = None
for stage, exp in [("pretrain", "capstone_pretrain"), ("sft", "capstone_sft"), ("joint", "capstone_joint"), ("dpo", "capstone_preference")]:
    path = ROOT / f"docs/course-experiments/results/{exp}.json"
    run = json.loads(path.read_text()); result = run["results"]
    input_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert run["seed"] == 42 and run["gpu"] == "NVIDIA L4" and run["torch_version"] == "2.14.1+cu126"
    assert result["data_manifest"] == manifest and result["test_evaluated"] is False and result["schedule_completed"]
    assert result["parent_checkpoint_sha256"] == prior_export
    prior_export = result["inference_export"]["sha256"]
    run_metadata[stage] = {"seed": run["seed"], "gpu": run["gpu"], "torch": run["torch_version"], "python": run["python_version"], "steps": result["steps"], "objective": result["objective"], "training_seconds": result["elapsed_training_seconds"], "experiment_seconds": run["elapsed_seconds"], "timing_scope": run["timing_scope"], "parent_sha256": result["parent_checkpoint_sha256"], "export_sha256": prior_export, "code_sha256": result["code_sha256"]}
code_hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in ["tiny_perceptron/capstone.py", "tiny_perceptron/data.py", "scripts/course_experiments/capstone.py"]}
result = {"environment": environment, "predictions": predictions, "inline_stdout": stdout.getvalue(), "split_fingerprints": manifest["sha256"], "input_sha256": input_hashes, "current_code_sha256": code_hashes, "selected_examples": [{k: row[k] for k in ("task", "user", "system", "answer")} for row in selected], "targets": target_counts, "random_dense_probe": {"logits_shape": list(logits.shape), "finite_loss": bool(loss.isfinite()), "loss": float(loss), "seed": 42, "updates": 0}, "protocol_cases": protocol_cases, "preference_pairs": pair_counts, "audit": audits, "constant_baselines": baselines, "original_run_metadata": run_metadata, "style_score_prerequisite": {"coefficient_0.2": [0.2 * 1 + 1, 0.2 * 3 + 0], "coefficient_2": [2 * 1 + 1, 2 * 3 + 0]}, "elapsed_seconds": time.perf_counter() - started, "limits": "Independent CPU validation of original stored GPU records and current small code; no own training, pretrained inference, GPU benchmark, weight download, or reproduction of L4 generations. General language, style diversity, danger understanding, calibrated uncertainty and web retrieval are untested by these templates."}
(OUT / "cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"environment": environment, "current_code_sha256": code_hashes, "targets": target_counts, "preference_pairs": pair_counts, "audited_records": 252, "sft": audits["sft"]["by_task"], "joint_text": 42, "dpo_text": 42, "dpo_joint": audits["dpo"]["by_task"]["joint"], "baselines": baselines, "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False, indent=2))
print("PASS: all bounded CPU assertions; zero optimizer updates.")
