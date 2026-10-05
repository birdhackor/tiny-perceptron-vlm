"""Bounded CPU verification of frozen historical evidence, never model evaluation."""
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
import tiny_perceptron.capstone as cap

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)


def read_original(relative):
    return json.loads((ART / "originals" / relative).read_bytes())


def check_trace(trace):
    assert cap.TOK.decode(trace["generated_ids"]) == trace["raw"]
    actual_eos = bool(trace["generated_ids"] and trace["generated_ids"][-1] == cap.TOK.eos_id)
    assert actual_eos == trace["eos"] == (trace["stop_reason"] == "eos")


data = read_original("docs/course-experiments/capstone-evidence/deployment/data.json")
split_report = {}
for split, rows in data["splits"].items():
    assert cap.digest(rows) == data["manifest"]["sha256"][split]
    assert len(rows) == data["manifest"]["counts"][split]
    fams = {r["family"] for r in rows if r["family"].startswith("numbers:")}
    assert fams == set(data["manifest"]["families"][split]["numbers"])
    for row in rows:
        if row["task"] == "calculator":
            a = cap.parse_action({"raw": row["answer"], "eos": True})
            assert row["user"] in (f"{a['a']}+{a['b']}等於多少？", f"請算{a['a']}加{a['b']}。")
    split_report[split] = {"rows": len(rows), "numeric_families": len(fams)}
for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not set(data["manifest"]["families"][a]["numbers"]) & set(data["manifest"]["families"][b]["numbers"])

stage_summary, selected_records, pointers = {}, {}, {}
for label, relative, split in [
    ("sft", "docs/course-experiments/capstone-evidence/sft/validation.json", "validation"),
    ("joint", "docs/course-experiments/capstone-evidence/joint/validation.json", "validation"),
    ("dpo", "docs/course-experiments/capstone-evidence/dpo/validation.json", "validation"),
    ("test-joint", "docs/course-experiments/capstone-evidence/deployment/test-joint.json", "test"),
]:
    raw = read_original(relative)
    rows_by_id = {row["id"]: row for row in data["splits"][split]}
    assert len(raw["records"]) == raw["count"] == len(rows_by_id)
    summaries, readable, read_pointers = {}, [], ["/count", "/protocol", "/by_task/calculator", "/by_task/unavailable", "/by_task/tool_return"]
    for index, record in enumerate(raw["records"]):
        if record["task"] not in ("calculator", "unavailable", "tool_return"):
            continue
        row = rows_by_id[record["id"]]
        assert record["family"] == row["family"]
        assert record["expected_action"] == row["answer"]
        assert record["expected_final"] == cap.expected_final(row)
        trace = record["action_trace"]
        check_trace(trace)
        assert trace["prompt_ids"] == cap.prompt_ids(row)
        action = cap.parse_action(trace)
        assert record["parsed_action"] == action
        action_correct = trace["eos"] and trace["raw"] == row["answer"]
        answer = None
        if action["status"] in ("direct", "ask"):
            answer = action["content"]
            assert record["runtime"] is None and record["final_trace"] is None
        elif action["status"] == "tool":
            expected_runtime = cap.calculator_runtime(action, available=row["available"])
            assert record["runtime"] == expected_runtime
            if expected_runtime["status"] == "ok":
                final = record["final_trace"]
                check_trace(final)
                followup = dict(row, image=None, audio=None)
                followup["user"] = f"原題：{action['a']}+{action['b']}。計算器回報：{expected_runtime['result']}。請回答。"
                assert final["prompt_ids"] == cap.prompt_ids(followup)
                parsed_final = cap.parse_action(final)
                if parsed_final["status"] == "direct":
                    answer = parsed_final["content"]
        end_correct = action_correct and answer == record["expected_final"]
        assert record["answer"] == answer
        assert record["action_correct"] == action_correct
        assert record["end_to_end_correct"] == end_correct
        out = summaries.setdefault(record["task"], {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
        out["count"] += 1
        out["action_correct"] += int(action_correct)
        out["end_to_end_correct"] += int(end_correct)
        fields = ["id", "family", "task", "expected_action", "expected_final", "action_trace", "parsed_action", "runtime", "final_trace", "answer", "action_correct", "end_to_end_correct"]
        read_pointers.extend(f"/records/{index}/{field}" for field in fields)
        readable.append({
            "pointer": f"/records/{index}", "id": record["id"], "question": row["user"],
            "first": trace["raw"], "first_eos": trace["eos"], "runtime": record["runtime"],
            "final": record["final_trace"]["raw"] if record["final_trace"] else None,
            "final_eos": record["final_trace"]["eos"] if record["final_trace"] else None,
            "answer": answer, "action_correct": action_correct, "end_to_end_correct": end_correct,
        })
    for task, computed in summaries.items():
        assert computed == raw["by_task"][task]
    stage_summary[label] = summaries
    selected_records[label] = readable
    pointers[relative] = read_pointers

assert stage_summary["test-joint"]["calculator"] == {"count": 12, "action_correct": 12, "end_to_end_correct": 10}
failures = [r for r in selected_records["test-joint"] if r["runtime"] and not r["end_to_end_correct"]]
assert [(r["question"], r["first"], r["runtime"]["result"], r["final"], r["final_eos"]) for r in failures] == [
    ("0+1等於多少？", "TOOL:calculator:0+1", "1", "DIRECT:0", True),
    ("請算1加0。", "TOOL:calculator:1+0", "1", "DIRECT:111", True),
]
example = next(r for r in selected_records["sft"] if r["question"] == "4+4等於多少？")
assert example["first"] == "TOOL:calculator:4+4" and example["runtime"]["result"] == "8"
assert example["final"] == "DIRECT:8" and example["first_eos"] and example["final_eos"]

# Control-flow witnesses use scripted generators, not trained-model capabilities.
boundary_results = []
for text, eos in [("TOOL:unknown:1+2", True), ("TOOL:calculator:1+2", False), ("TOOL:calculator:1000+1", True), ("TOOL:calculator:-1+2", True), ("TOOL:calculator:999+999", True), ("TOOL:calculator:1+2;print(1)", True)]:
    action = cap.parse_action({"raw": text, "eos": eos})
    boundary_results.append({"raw": text, "eos": eos, "action": action, "runtime": cap.calculator_runtime(action)})
assert boundary_results[0]["action"]["status"] == "tool"
assert boundary_results[0]["runtime"]["reason"] == "tool_not_allowlisted"
assert boundary_results[1]["action"]["reason"] == "unterminated_generation"
assert boundary_results[2]["action"]["status"] == "invalid"
assert boundary_results[4]["runtime"] == {"status": "ok", "result": "1998"}
assert boundary_results[5]["action"]["status"] == "invalid"
bool_argument = cap.calculator_runtime({"status": "tool", "name": "calculator", "a": True, "b": 2})
assert bool_argument["reason"] == "invalid_arguments"

row = next(r for r in data["splits"]["test"] if r["user"] == "1+2等於多少？" and r["task"] == "calculator")
identity = object()
flow_witnesses = []
for available, replacement in [(True, None), (True, "9"), (False, None)]:
    calls = []
    def generator(model, current, max_new_tokens):
        assert model is identity
        calls.append(current["user"])
        answer = "TOOL:calculator:1+2" if len(calls) == 1 else ("DIRECT:9" if replacement else "DIRECT:3")
        return {"raw": answer, "eos": True}
    def runtime(action, available):
        return {"status": "ok", "result": replacement} if replacement else cap.calculator_runtime(action, available)
    with patch.object(cap, "generate_trace", generator):
        result = cap.run_assistant(identity, dict(row, available=available), runtime=runtime)
    assert len(calls) == (2 if available else 1)
    if available:
        returned = replacement or "3"
        assert calls[1] == f"原題：1+2。計算器回報：{returned}。請回答。"
        assert result["answer"] == returned and result["final_trace"] is not None
    else:
        assert result["answer"] is None and result["runtime"]["reason"] == "calculator_unavailable"
    flow_witnesses.append({"available": available, "test_replacement": replacement, "generator_inputs": calls, "record": result})

result = {
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "cuda_build": str(torch.version.cuda)},
    "data_split_verification": split_report,
    "recomputed_tool_summaries": stage_summary,
    "selected_records": selected_records,
    "read_pointers": pointers,
    "historical_failures": failures,
    "boundaries": boundary_results,
    "boolean_argument": bool_argument,
    "scripted_control_flow_witnesses": flow_witnesses,
    "limits": "Historical traces were decoded and recalculated. Scripted generators verify control flow only; no weights loaded, no new model score, no training.",
}
(ART / "verification-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"splits": split_report, "summaries": stage_summary, "failures": failures, "bounded_checks": "passed"}, ensure_ascii=False, indent=2))
