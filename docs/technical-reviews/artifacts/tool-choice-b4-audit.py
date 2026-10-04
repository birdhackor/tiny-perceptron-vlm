"""Independent B.4 audit: saved model evidence and local protocol fixtures.

The fixture sampler is deliberately scripted: it validates controller behavior,
not model quality. Recorded L4 generations remain the empirical source.
"""

import copy
import hashlib
import json
import math
import platform
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.course_experiments import applications as app
from scripts.course_experiments.common import records_sha256, split_records
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.retrieval import tool_loop

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "docs/course-experiments/results/tools.json"
document = json.loads(REPORT.read_text())
results = document["results"]
tokenizer = ByteTokenizer()


def prompt_ids(messages):
    roles = {"system": 7, "user": 3, "assistant": 4}
    answer = [1]
    for message in messages:
        answer += [roles[message["role"]]] + [b + 8 for b in message["content"].encode()] + [2]
    return answer + [4]


def audit_episode(episode):
    question = episode["question"]
    if episode["operation"] == "copy":
        expected = int(question.removeprefix("COPY:"))
        desired = None
    else:
        operation, a, b = re.fullmatch(r"CALC:(add|multiply)\((\d+),(\d+)\)", question).groups()
        a, b = int(a), int(b)
        expected = a + b if operation == "add" else a * b
        desired = {"name": operation, "arguments": {"a": a, "b": b}}
    assert expected == episode["expected"]
    messages = [{"role": "system", "content": app._TOOLS_SYSTEM}, {"role": "user", "content": question}]
    calls, generations, mismatched_messages, errors = [], [], [], []
    answer = None
    terminal = "step_limit"
    for event in episode["trace"]:
        generation = event["generation"]
        sample = generation["samples"][0]
        assert generation["input_ids"] == prompt_ids(messages)
        assert generation["input_tokens"] == len(generation["input_ids"])
        if generation["messages"] != messages:
            mismatched_messages.append(event["step"])
        ids = sample["generated_ids"]
        eos = bool(ids and ids[-1] == 2)
        raw_ids = ids[:-1] if eos else ids
        illegal = [i for i in raw_ids if i < 8]
        text = bytes(i - 8 for i in raw_ids if i >= 8).decode("utf-8", errors="replace")
        assert text == sample["generated"]
        assert eos == sample["eos"] and illegal == sample["invalid_special_tokens"]
        assert sample["generated_tokens"] == generation["generated_tokens"] == len(ids)
        assert 0 <= generation["seconds"] <= episode["seconds"]
        generations.append({"step": event["step"], "eos": eos, "illegal": illegal, "tokens": len(ids)})
        try:
            action = json.loads(text)
        except ValueError:
            assert not event["executed"] and "parsed" not in event
            terminal = "invalid_request"
            errors.append(text)
            continue
        assert isinstance(action, dict) and action == event["parsed"]
        if "done" in action:
            assert set(action) == {"done", "answer"} and action["done"] is True
            answer, terminal = action["answer"], "done"
            assert not event["executed"]
        else:
            assert set(action) == {"name", "arguments"} and action["name"] in {"add", "multiply"}
            args = action["arguments"]
            assert set(args) == {"a", "b"} and all(type(v) in (int, float) for v in args.values())
            value = args["a"] + args["b"] if action["name"] == "add" else args["a"] * args["b"]
            matches = action == desired
            assert event["executed"] and math.isfinite(value)
            assert event["tool_result"] == value and event["finite_result"] is True
            assert event["request_matches_question"] == matches
            calls.append({"value": value, "matches": matches})
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": f"TOOL_RESULT:{value}"}]
    answer_correct = terminal == "done" and type(answer) in (int, float) and answer == expected
    behavior = not calls if desired is None else bool(calls and any(c["matches"] for c in calls) and answer == calls[-1]["value"])
    success = answer_correct and behavior
    assert episode["status"] == terminal and episode["answer"] == answer
    assert episode["answer_correct"] == answer_correct
    assert episode["required_tool_behavior"] == behavior and episode["correct"] == success
    assert episode["actual_tool_calls"] == len(calls)
    assert episode["generated_tokens"] == sum(g["tokens"] for g in generations)
    assert episode["seconds"] >= sum(event["generation"]["seconds"] for event in episode["trace"])
    return {
        "question": question, "operation": episode["operation"], "max_steps": episode["max_steps"],
        "status": terminal, "answer": answer, "expected": expected, "correct": success,
        "generations": generations, "actual_calls": len(calls), "correct_calls": sum(c["matches"] for c in calls),
        "input_ids_match_reconstructed_input": True, "messages_snapshot_mismatch_steps": mismatched_messages,
        "episode_seconds": episode["seconds"],
        "generation_seconds_sum": sum(event["generation"]["seconds"] for event in episode["trace"]),
        "parse_error_outputs": errors,
    }


normal = [audit_episode(row) for row in results["samples"]]
capped = [audit_episode(row) for row in results["one_step_cap_samples"]]
assert len(normal) == 20 and len(capped) == 18
assert {row["question"] for row in capped} == {row["question"] for row in normal if row["operation"] != "copy"}
assert all(row["max_steps"] == 3 for row in normal)
assert all(row["status"] == "step_limit" and row["actual_calls"] == 1 and len(row["generations"]) == 1 for row in capped)
all_generations = [g for row in normal + capped for g in row["generations"]]
summary = {
    "normal_tasks": len(normal), "normal_arithmetic_tasks": sum(r["operation"] != "copy" for r in normal),
    "normal_copy_tasks": sum(r["operation"] == "copy" for r in normal),
    "normal_first_action_objects": sum(isinstance(json.loads(row["trace"][0]["generation"]["samples"][0]["generated"]), dict) for row in results["samples"]),
    "normal_done": sum(r["status"] == "done" for r in normal),
    "normal_correct_answers": sum(row["answer_correct"] for row in results["samples"]),
    "normal_successes": sum(r["correct"] for r in normal), "normal_status_counts": dict(Counter(r["status"] for r in normal)),
    "normal_calls": sum(r["actual_calls"] for r in normal), "normal_correct_calls": sum(r["correct_calls"] for r in normal),
    "capped_tasks": len(capped), "capped_calls": sum(r["actual_calls"] for r in capped),
    "capped_status_counts": dict(Counter(r["status"] for r in capped)),
    "normal_generations": sum(len(r["generations"]) for r in normal), "capped_generations": sum(len(r["generations"]) for r in capped),
    "all_generations": len(all_generations), "eos_generations": sum(g["eos"] for g in all_generations),
    "illegal_special_tokens": sum(len(g["illegal"]) for g in all_generations),
    "normal_generated_tokens": sum(g["tokens"] for r in normal for g in r["generations"]),
    "capped_generated_tokens": sum(g["tokens"] for r in capped for g in r["generations"]),
    "all_generated_tokens_including_eos": sum(g["tokens"] for g in all_generations),
    "text_byte_tokens_excluding_eos": sum(g["tokens"] - int(g["eos"]) for g in all_generations),
    "message_snapshot_mismatches": sum(len(r["messages_snapshot_mismatch_steps"]) for r in normal + capped),
    "input_ids_verified": len(all_generations),
}
assert summary["all_generations"] == summary["eos_generations"] == 56
assert summary["all_generated_tokens_including_eos"] == results["generated_tokens"] == 2072
assert summary["normal_done"] == summary["normal_correct_answers"] == summary["normal_successes"] == 19
assert summary["normal_calls"] == summary["normal_correct_calls"] == 18
for key, count in [("accuracy", 19), ("answer_accuracy", 19), ("completion_rate", 19), ("first_action_json_rate", 20)]:
    assert results[key] == {"numerator": count, "denominator": 20, "rate": count / 20}
assert results["correct_parameters_given_executed_call"] == {"numerator": 18, "denominator": 18, "rate": 1.0}
assert results["status_counts"] == summary["normal_status_counts"]

splits = split_records(app._tool_records(), 42)
assert results["split"] == {key: {"records": len(rows), "families": len({r["family"] for r in rows}), "sha256": records_sha256(rows)} for key, rows in splits.items()}
assert [r["messages"][1]["content"] for r in splits["test"]] == [r["question"] for r in results["samples"]]
family_sets = [set(r["family"] for r in rows) for rows in splits.values()]
assert all(not family_sets[i] & family_sets[j] for i in range(3) for j in range(i + 1, 3))

requests = [{"name": "add", "arguments": {"a": 2, "b": 3}}, {"done": True, "answer": 5}]
local = {
    "baseline": tool_loop(requests, max_steps=3), "capped": tool_loop(requests, max_steps=1),
    "exhausted": tool_loop(requests[:1], max_steps=3),
    "invalid": tool_loop([{"name": "unknown", "arguments": {"a": 2, "b": 3}}]),
    "wrong_done": tool_loop([requests[0], {"done": True, "answer": 6}], max_steps=3),
    "exercise_cap_two": tool_loop(requests, max_steps=2),
}
assert [(r["status"], len(r["trace"]), r.get("answer")) for r in list(local.values())[:4]] == [("done", 1, 5), ("step_limit", 1, None), ("needs_more_model_output", 1, None), ("invalid_request", 0, None)]
assert local["wrong_done"]["status"] == "done" and local["wrong_done"]["answer"] == 6
assert local["exercise_cap_two"] == local["baseline"]


def fixture_episode(operation, expected, texts, max_steps=3):
    record = {"family": "fixture", "operation": operation, "answer": expected,
              "messages": [{"role": "system", "content": app._TOOLS_SYSTEM}, {"role": "user", "content": "fixture"}]}
    if operation != "copy": record.update(a=2, b=3)
    iterator = iter(texts)
    observed_inputs = []
    def sampler(model, messages, ctx, tokens):
        observed_inputs.append(copy.deepcopy(messages))
        text = next(iterator)
        return {"messages": copy.deepcopy(messages), "samples": [{"generated": text, "invalid_special_tokens": []}], "generated_tokens": len(text) + 1, "seconds": 0.0}
    with patch.object(app, "_sample", sampler):
        episode = app._tool_episode(object(), record, SimpleNamespace(device="cpu"), max_steps=max_steps)
    return {"status": episode["status"], "answer": episode["answer"], "answer_correct": episode["answer_correct"],
            "required_tool_behavior": episode["required_tool_behavior"], "correct": episode["correct"],
            "actual_tool_calls": episode["actual_tool_calls"], "sampler_inputs": observed_inputs}


call_text = '{"name":"add","arguments":{"a":2,"b":3}}'
fixtures = {
    "direct_correct_without_required_call": fixture_episode("add", 5, ['{"done":true,"answer":5}']),
    "wrong_answer_after_valid_call": fixture_episode("add", 5, [call_text, '{"done":true,"answer":6}']),
    "valid_call_and_completion": fixture_episode("add", 5, [call_text, '{"done":true,"answer":5}']),
    "valid_call_one_step": fixture_episode("add", 5, [call_text], max_steps=1),
    "copy_with_unneeded_call": fixture_episode("copy", 5, [call_text, '{"done":true,"answer":5}']),
}
assert fixtures["direct_correct_without_required_call"]["answer_correct"] and not fixtures["direct_correct_without_required_call"]["correct"]
assert fixtures["valid_call_and_completion"]["correct"]
assert not fixtures["wrong_answer_after_valid_call"]["correct"] and not fixtures["copy_with_unneeded_call"]["correct"]
assert fixtures["valid_call_one_step"]["status"] == "step_limit"
assert fixtures["valid_call_and_completion"]["sampler_inputs"][1][-1] == {"role": "user", "content": "TOOL_RESULT:5"}
unknown = app._parse_json_action('{"name":"unknown","arguments":{"a":2,"b":3}}')
assert isinstance(unknown, dict)

code_hashes = {}
for name in ["tiny_perceptron/retrieval.py", "tiny_perceptron/data.py", "scripts/course_experiments/applications.py", "scripts/course_experiments/common.py"]:
    digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert digest == document["code_sha256"][name]
    code_hashes[name] = digest

print(json.dumps({
    "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/tool-choice-b4-audit.py",
    "audit_environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu; no model training or sampling performed"},
    "recorded_experiment_environment": {key: document[key] for key in ["revision", "python_version", "torch_version", "device", "gpu", "seed", "evidence_status"]},
    "original_report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(), "matching_recorded_code_sha256": code_hashes,
    "split": results["split"], "summary": summary, "normal_episodes": normal, "capped_episodes": capped,
    "local_protocol_executions": local, "controller_fixture_executions": fixtures,
    "evidence_limit": "36 generation.messages fields share a mutable conversation list and contain future TOOL_RESULT messages. All 56 original input_ids equal independently reconstructed pre-generation inputs. Counts and parameter checks use raw generated_ids/text, not those mutable message fields. 2072 generated tokens include 56 EOS and 2016 text byte tokens.",
    "result": "All independent assertions passed; original L4 results are corroborated, not rerun on CPU."
}, ensure_ascii=False, indent=2))
