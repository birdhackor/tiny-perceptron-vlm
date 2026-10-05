"""Bounded CPU checks and replay of immutable recorded generations; no model inference/training."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.retrieval import call_tool, tool_loop
from scripts.course_experiments import applications as app
from scripts.course_experiments.common import records_sha256, split_records

assert torch.version.cuda is None and not torch.cuda.is_available()
tok = ByteTokenizer()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

requests = [{"name": "add", "arguments": {"a": 2, "b": 3}}, {"done": True, "answer": 5}]
original = [tool_loop(requests, max_steps=3), tool_loop(requests, max_steps=1),
            tool_loop(requests[:1], max_steps=3), tool_loop([{"name": "unknown", "arguments": {"a": 2, "b": 3}}])]
assert [x["status"] for x in original] == ["done", "step_limit", "needs_more_model_output", "invalid_request"]
assert [len(x["trace"]) for x in original] == [1, 1, 1, 0]
assert [x.get("answer") for x in original] == [5, None, None, None]
two_steps = tool_loop(requests, max_steps=2)
wrong_answer = tool_loop([requests[0], {"done": True, "answer": 6}], max_steps=2)
assert two_steps["status"] == "done" and two_steps["answer"] == 5 and len(two_steps["trace"]) == 1
assert wrong_answer["status"] == "done" and wrong_answer["answer"] == 6 and wrong_answer["answer"] != wrong_answer["trace"][0]["result"]
exhausted_at_budget = tool_loop(requests[:1], max_steps=1)
assert exhausted_at_budget["status"] == "needs_more_model_output"

raw_path = BASE / "inputs/docs/course-experiments/results/tools.json"
raw = json.loads(raw_path.read_bytes())
results = raw["results"]
normal, capped = results["samples"], results["one_step_cap_samples"]
assert len(normal) == 20 and len(capped) == 18
code_matches = {}
for relative in ["tiny_perceptron/retrieval.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/applications.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py"]:
    code_matches[relative] = sha(ROOT / relative) == raw["code_sha256"][relative] == sha(BASE / "inputs" / relative)
assert all(code_matches.values())

records = app._tool_records()
splits = split_records(records, raw["seed"])
record_by_question = {r["messages"][1]["content"]: r for r in records}
assert [r["messages"][1]["content"] for r in splits["test"]] == [e["question"] for e in normal]
split_hashes = {name: records_sha256(rows) for name, rows in splits.items()}
for name, rows in splits.items():
    assert split_hashes[name] == results["split"][name]["sha256"]
    assert len(rows) == results["split"][name]["records"]
    assert len({r["family"] for r in rows}) == results["split"][name]["families"]
families = {name: {r["family"] for r in rows} for name, rows in splits.items()}
assert not families["train"] & families["validation"]
assert not families["train"] & families["test"]
assert not families["validation"] & families["test"]

def strict_object(text):
    def reject(value):
        raise ValueError(value)
    def float_value(text):
        number = float(text)
        if not math.isfinite(number):
            raise ValueError(text)
        return number
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(key)
            value[key] = item
        return value
    value = json.loads(text, parse_constant=reject, parse_float=float_value, object_pairs_hook=unique)
    assert type(value) is dict
    return value

def question_parts(question):
    if question.startswith("COPY:"):
        return "copy", int(question[5:]), None
    m = re.fullmatch(r"CALC:(add|multiply)\((\d),(\d)\)", question)
    assert m
    operation, a, b = m[1], int(m[2]), int(m[3])
    return operation, a + b if operation == "add" else a * b, {"name": operation, "arguments": {"a": a, "b": b}}

generation_rows, episode_rows = [], []
for group, episodes in [("samples", normal), ("one_step_cap_samples", capped)]:
    for index, e in enumerate(episodes):
        operation, expected, expected_request = question_parts(e["question"])
        assert e["operation"] == operation and e["expected"] == expected
        record = record_by_question[e["question"]]
        messages = copy.deepcopy(record["messages"][:2])
        executions = []
        parsed_first = False
        done_answer = None
        independently_status = "step_limit"
        for step, event in enumerate(e["trace"]):
            g = event["generation"]
            assert g["candidate_count"] == 1 and g["max_new_tokens"] == 64 and g["temperature"] == 0.0
            assert g["input_ids"] == app._prompt_ids(messages, tok)
            assert g["input_tokens"] == len(g["input_ids"])
            s = g["samples"][0]
            ids = s["generated_ids"]
            independently_eos = bool(ids and ids[-1] == tok.eos_id)
            content_ids = ids[:-1] if independently_eos else ids
            illegal = [t for t in content_ids if t < 8]
            assert independently_eos == s["eos"] and illegal == s["invalid_special_tokens"]
            assert tok.decode(content_ids) == s["generated"]
            assert s["generated_tokens"] == len(ids) == g["generated_tokens"]
            assert 0 < g["seconds"] <= e["seconds"]
            generation_rows.append({"pointer": f"/results/{group}/{index}/trace/{step}/generation/samples/0", "eos_from_ids": independently_eos,
                                    "illegal_control_tokens": illegal, "tokens": len(ids), "generation_seconds": g["seconds"]})
            try:
                if illegal:
                    raise ValueError("control token")
                action = strict_object(s["generated"])
                if step == 0:
                    parsed_first = True
                if "done" in action:
                    assert set(action) == {"done", "answer"} and action["done"] is True
                    assert type(action["answer"]) in (int, float, str)
                    done_answer = action["answer"]
                    independently_status = "done"
                    assert not event["executed"]
                    break
                assert set(action) == {"name", "arguments"} and action["name"] in ("add", "multiply")
                assert type(action["arguments"]) is dict and set(action["arguments"]) == {"a", "b"}
                assert all(type(v) in (int, float) for v in action["arguments"].values())
                a, b = action["arguments"]["a"], action["arguments"]["b"]
                actual_result = a + b if action["name"] == "add" else a * b
                assert event["executed"] and event["tool_result"] == actual_result and event["finite_result"]
                assert event["request_matches_question"] == (operation != "copy" and action == expected_request)
                assert call_tool(action) == actual_result
                executions.append({"action": action, "result": actual_result, "matches": action == expected_request})
                messages += [{"role": "assistant", "content": s["generated"]}, {"role": "user", "content": f"TOOL_RESULT:{actual_result}"}]
            except (ValueError, TypeError, KeyError):
                independently_status = "invalid_request"
                assert not event["executed"]
                break
        answer_correct = independently_status == "done" and type(done_answer) in (int, float) and done_answer == expected
        required = not executions if operation == "copy" else bool(executions and any(x["matches"] for x in executions) and done_answer == executions[-1]["result"])
        correct = answer_correct and required
        assert (independently_status, done_answer, answer_correct, required, correct, len(executions)) == (e["status"], e["answer"], e["answer_correct"], e["required_tool_behavior"], e["correct"], e["actual_tool_calls"])
        assert e["generated_tokens"] == sum(x["generation"]["generated_tokens"] for x in e["trace"])
        episode_rows.append({"pointer": f"/results/{group}/{index}", "question": e["question"], "status": independently_status,
                             "answer": done_answer, "expected_from_question": expected, "correct_from_criteria": correct,
                             "executed_calls": len(executions), "first_action_object": parsed_first,
                             "generation_count": len(e["trace"]), "episode_seconds": e["seconds"]})

normal_rows = episode_rows[:20]
capped_rows = episode_rows[20:]
assert Counter(e["operation"] for e in normal) == {"add": 9, "multiply": 9, "copy": 2}
assert all(e["max_steps"] == 3 for e in normal) and all(e["max_steps"] == 1 for e in capped)
assert [e["question"] for e in capped] == [e["question"] for e in normal if e["operation"] != "copy"]
assert all(x["status"] == "step_limit" and x["executed_calls"] == 1 and x["generation_count"] == 1 for x in capped_rows)
normal_calls = sum(x["executed_calls"] for x in normal_rows)
correct_calls = sum(x["request_matches_question"] for e in normal for x in e["trace"] if x["executed"])
assert normal_calls == correct_calls == results["actual_tool_calls"] == 18
rates = {
    "accuracy": sum(x["correct_from_criteria"] for x in normal_rows),
    "answer_accuracy": sum(e["answer_correct"] for e in normal),
    "completion_rate": sum(x["status"] == "done" for x in normal_rows),
    "first_action_json_rate": sum(x["first_action_object"] for x in normal_rows),
}
for name, numerator in rates.items():
    assert results[name] == {"numerator": numerator, "denominator": 20, "rate": numerator / 20}
assert results["correct_parameters_given_executed_call"] == {"numerator": 18, "denominator": 18, "rate": 1.0}
assert results["status_counts"] == dict(Counter(x["status"] for x in normal_rows))
assert sum(len(e["trace"]) for e in normal if e["operation"] != "copy") == 36
assert sum(len(e["trace"]) for e in normal if e["operation"] == "copy") == 2
assert len(generation_rows) == 36 + 2 + 18 == 56
assert sum(x["eos_from_ids"] for x in generation_rows) == 56 and not any(x["illegal_control_tokens"] for x in generation_rows)
assert sum(x["tokens"] for x in generation_rows) == results["generated_tokens"] == 2072

# Re-run only protocol control flow with recorded generations; never call a model.
original_sample = app._sample
replayed = 0
try:
    for e in normal + capped:
        cursor = 0
        def recorded_sample(model, messages, ctx, tokens):
            global cursor
            assert model is None and tokens == 64
            g = copy.deepcopy(e["trace"][cursor]["generation"])
            assert g["input_ids"] == app._prompt_ids(messages, tok)
            cursor += 1
            return g
        app._sample = recorded_sample
        replay = app._tool_episode(None, record_by_question[e["question"]], SimpleNamespace(device="cpu"), max_steps=e["max_steps"])
        assert cursor == len(e["trace"])
        for key in ["status", "answer", "answer_correct", "required_tool_behavior", "correct", "actual_tool_calls", "generated_tokens"]:
            assert replay[key] == e[key], (e["question"], key)
        replayed += 1
finally:
    app._sample = original_sample
assert replayed == 38

summary = {
    "source_sha256": sha(BASE / "inputs/section.md"),
    "input_tools_sha256": sha(raw_path),
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "cuda_available": str(torch.cuda.is_available())},
    "original_four_statuses": [x["status"] for x in original], "original_call_counts": [len(x["trace"]) for x in original],
    "max_steps_two": two_steps, "wrong_answer_still_done": wrong_answer,
    "list_exhaustion_at_budget": exhausted_at_budget,
    "normal_task_success": [rates["accuracy"], 20], "normal_first_action_objects": [rates["first_action_json_rate"], 20],
    "normal_exact_question_tool_calls": [correct_calls, normal_calls],
    "normal_statuses": dict(Counter(x["status"] for x in normal_rows)), "one_step_statuses": dict(Counter(x["status"] for x in capped_rows)),
    "generation_denominator": {"normal_arithmetic": 36, "normal_copy": 2, "one_step_arithmetic": 18, "total": 56},
    "eos_from_original_ids": [sum(x["eos_from_ids"] for x in generation_rows), len(generation_rows)],
    "illegal_control_token_generations": sum(bool(x["illegal_control_tokens"]) for x in generation_rows),
    "generated_tokens": sum(x["tokens"] for x in generation_rows), "trace_replay_episodes": replayed,
    "replay_method": "Original _tool_episode; _sample supplies immutable recorded generation dicts; model=None; no inference or training.",
    "split_counts": {k: len(v) for k, v in splits.items()}, "split_hashes": split_hashes, "code_matches_measured_revision": code_matches,
    "inspected_json_pointers": ["/schema_version", "/experiment_id", "/revision", "/device", "/seed", "/torch_version", "/python_version", "/gpu", "/step_scale", "/artifacts", "/code_sha256", "/results/protocol/roles", "/results/protocol/external_result_serialization", "/results/protocol/allowlist", "/results/protocol/final", "/results/protocol/max_steps_including_done", "/results/split", "/results/samples", "/results/one_step_cap_samples", "/results/accuracy", "/results/answer_accuracy", "/results/completion_rate", "/results/first_action_json_rate", "/results/correct_parameters_given_executed_call", "/results/status_counts", "/results/actual_tool_calls", "/results/generated_tokens"],
    "generation_rows": generation_rows, "episode_rows": episode_rows,
}
(BASE / "checks/bounded-result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k not in ["generation_rows", "episode_rows", "inspected_json_pointers"]}, indent=2, ensure_ascii=False))
