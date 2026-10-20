"""獨立核對課文算例、工具協議與保存的逐步輸入。"""

import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.applications import (
    _TOOLS_SYSTEM,
    _parse_json_action,
    _prompt_ids,
    _tool_records,
)
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.retrieval import call_tool


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def reconstruct(ids):
    tok = ByteTokenizer()
    roles = {tok.system_id: "system", tok.user_id: "user", tok.assistant_id: "assistant"}
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.assistant_id
    index, messages = 1, []
    while index < len(ids) - 1:
        role = roles[ids[index]]
        end = ids.index(tok.eos_id, index + 1)
        raw = ids[index + 1 : end]
        assert all(value >= 8 for value in raw)
        messages.append({"role": role, "content": tok.decode(raw)})
        index = end + 1
    assert index == len(ids) - 1
    assert _prompt_ids(messages, tok) == ids
    return messages


tok = ByteTokenizer()
request = {"name": "multiply", "arguments": {"a": 123, "b": 45}}
result = call_tool(request)
assert result == 5535
assert call_tool(json.dumps(request)) == result
trace = [{"role": "assistant", "tool_request": request}, {"role": "tool", "result": result}]
final = {"role": "assistant", "content": '{"done":true,"answer":5535}'}
roles = {}
for role in ("tool", "user", "system", "assistant"):
    try:
        x, y = render_chat([{"role": role, "content": f"TOOL_RESULT:{result}"}, final])
        roles[role] = {"accepted": True, "input_length": len(x), "target_length": len(y)}
    except ValueError as error:
        roles[role] = {"accepted": False, "error": str(error)}
assert roles["tool"]["accepted"] is False
assert roles["user"]["input_length"] == 47
replacement = call_tool({"name": "multiply", "arguments": {"a": 123, "b": 46}})
assert replacement == 5658
assert final["content"] == '{"done":true,"answer":5535}'
replacement_trace = [
    {"role": "assistant", "tool_request": {"name": "multiply", "arguments": {"a": 123, "b": 46}}},
    {"role": "tool", "result": replacement},
]

path = "docs/course-experiments/results/tools.json"
raw = json.loads((ROOT / path).read_bytes())
samples = raw["results"]["samples"]
arithmetic, copies, executions, snapshots = [], [], [], []
for population in ("samples", "one_step_cap_samples"):
    for sample_index, sample in enumerate(raw["results"][population]):
        question = sample["question"]
        for event_index, event in enumerate(sample["trace"]):
            if "generation" not in event:
                continue
            gen = event["generation"]
            messages = reconstruct(gen["input_ids"])
            assert messages[0] == {"role": "system", "content": _TOOLS_SYSTEM}
            assert messages[1] == {"role": "user", "content": question}
            generated = gen["samples"][0]
            assert tok.decode(generated["generated_ids"]) == generated["generated"]
            assert generated["eos"] == (generated["generated_ids"][-1] == tok.eos_id)
            assert len(gen["input_ids"]) == gen["input_tokens"]
            assert len(generated["generated_ids"]) == generated["generated_tokens"]
            if event_index:
                previous = sample["trace"][event_index - 1]
                assert previous["executed"]
                assert messages[-2] == {
                    "role": "assistant", "content": previous["generation"]["samples"][0]["generated"]
                }
                assert messages[-1] == {"role": "user", "content": f"TOOL_RESULT:{previous['tool_result']}"}
            snapshots.append({
                "original_json_pointer": f"/results/{population}/{sample_index}/trace/{event_index}/generation",
                "question": question,
                "messages_at_generation": messages,
                "input_ids_count": len(gen["input_ids"]),
                "reencoded_equal": True,
                "stored_messages_equal": gen["messages"] == messages,
                "generated": generated["generated"],
            })
        if population != "samples":
            continue
        calls = [event for event in sample["trace"] if event.get("executed")]
        assert sample["actual_tool_calls"] == len(calls)
        if sample["operation"] == "copy":
            assert not calls and sample["status"] == "done"
            assert sample["answer"] == int(question.split(":")[1])
            copies.append({"question": question, "answer": sample["answer"], "calls": 0})
        else:
            assert len(calls) == 1
            match = re.fullmatch(r"CALC:(add|multiply)\((\d+),(\d+)\)", question)
            operation, a, b = match.group(1), int(match.group(2)), int(match.group(3))
            expected_request = {"name": operation, "arguments": {"a": a, "b": b}}
            action = _parse_json_action(calls[0]["generation"]["samples"][0]["generated"])
            assert action == calls[0]["parsed"] == expected_request
            observed = call_tool(action)
            assert observed == calls[0]["tool_result"]
            assert math.isfinite(observed) and calls[0]["finite_result"]
            assert calls[0]["request_matches_question"]
            executions.append(observed)
            arithmetic.append({"question": question, "calls": len(calls), "request": action,
                               "result": observed, "status": sample["status"],
                               "final_generated": sample["trace"][-1]["generation"]["samples"][0]["generated"]})
assert len(samples) == 20 and len(arithmetic) == len(executions) == 18 and len(copies) == 2
addition = next(row for row in arithmetic if row["question"] == "CALC:add(9,9)")
assert addition["result"] == 18 and addition["final_generated"] == '{"done":true,"answer":18}'
failure = next(row for row in arithmetic if row["question"] == "CALC:multiply(9,9)")
assert failure["result"] == 81 and failure["final_generated"] == '{"done":true,"answer":8}}'
try:
    _parse_json_action(failure["final_generated"])
    raise AssertionError("不合法的完成訊息不應解析成功")
except ValueError as error:
    parser_error = str(error)
assert failure["status"] == "invalid_request"
assert len(snapshots) == 56

records = _tool_records()
training_arithmetic = [r for r in records if r["operation"] != "copy"]
for row in training_arithmetic:
    assert row["messages"][0] == {"role": "system", "content": _TOOLS_SYSTEM}
    assert row["messages"][3] == {"role": "user", "content": f"TOOL_RESULT:{row['answer']}"}
assert len(records) == 210 and len(training_arithmetic) == 200

historical_command = ["git", "show", raw["revision"] + ":tiny_perceptron/data.py"]
historical = subprocess.run(historical_command, capture_output=True, text=True)
assert sha("tiny_perceptron/data.py") == raw["code_sha256"]["tiny_perceptron/data.py"]
assert sha("tiny_perceptron/retrieval.py") == raw["code_sha256"]["tiny_perceptron/retrieval.py"]
source_paths = [path, "tiny_perceptron/data.py", "tiny_perceptron/retrieval.py",
                "scripts/course_experiments/applications.py", "course/figures/rewrite-B-tool-flow.svg"]
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-B-3/verify-facts.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "source_sha256": {name: sha(name) for name in source_paths},
    "arithmetic": {"multiply_123_45": result, "multiply_123_46": replacement, "trace": trace,
                   "replacement_trace": replacement_trace, "new_feedback": f"TOOL_RESULT:{replacement}",
                   "handwritten_final_after_replacement": final,
                   "chat_length_terms": {"bos": 1, "roles": 2, "message_eos": 2,
                                         "feedback_bytes": len(f"TOOL_RESULT:{result}".encode()),
                                         "answer_bytes": len(final["content"].encode()), "shift_drop": 1}},
    "roles": roles,
    "experiment_provenance": {"revision": raw["revision"], "seed": raw["seed"],
                              "python": raw["python_version"], "torch": raw["torch_version"],
                              "device": raw["device"], "gpu": raw["gpu"],
                              "training_steps": raw["results"]["training"]["steps"],
                              "split": raw["results"]["split"],
                              "recorded_code_sha256": {n: raw["code_sha256"][n] for n in source_paths if n in raw["code_sha256"]},
                              "historical_lookup": {"command": historical_command, "exit_code": historical.returncode,
                                                    "stdout": historical.stdout, "stderr": historical.stderr}},
    "population": {"test": len(samples), "arithmetic": len(arithmetic), "calls": len(executions),
                   "matching_requests": len(arithmetic), "finite_results": len(executions),
                   "copies": len(copies), "reencoded_snapshots": len(snapshots),
                   "stored_messages_mismatch": sum(not r["stored_messages_equal"] for r in snapshots),
                   "demonstration_records": len(records), "arithmetic_demonstrations": len(training_arithmetic)},
    "arithmetic_samples": arithmetic,
    "copy_samples": copies,
    "invalid_final_parser_error": parser_error,
    "independent_snapshots": snapshots,
    "scope": "重新運算保存的工具請求、重建生成輸入並核對當前程式；未重新訓練或重新生成保存的模型輸出。",
    "result": "所有 assertion 通過。",
}, ensure_ascii=False, indent=2))
