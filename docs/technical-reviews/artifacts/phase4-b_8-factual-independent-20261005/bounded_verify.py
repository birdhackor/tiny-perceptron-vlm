"""Independent B.8 CPU arithmetic and replay of existing generated text; no model runs."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import re
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


data_path = HERE / "frozen-input/docs/course-experiments/results/tools.json"
applications_path = HERE / "frozen-input/scripts/course_experiments/applications.py"
retrieval_path = HERE / "frozen-input/tiny_perceptron/retrieval.py"
data = json.loads(data_path.read_bytes())
for relative, frozen in [
    ("scripts/course_experiments/applications.py", applications_path),
    ("tiny_perceptron/retrieval.py", retrieval_path),
]:
    assert sha(ROOT / relative) == sha(frozen) == data["code_sha256"][relative]
assert sha(ROOT / "docs/course-experiments/results/tools.json") == sha(data_path)

# Execute the exact original fence with only the one requested probability changed.
fence = (HERE / "original-fence/fence-1.py").read_bytes()
variant = ast.parse(fence)
assignment = next(n for n in variant.body if isinstance(n, ast.Assign) and any(
    isinstance(t, ast.Name) and t.id == "direct_accuracy" for t in n.targets))
assert assignment.value.value == 0.90
assignment.value.value = 0.999
namespace = {}
stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    exec(compile(variant, "B.8-original-fence-direct-0.999", "exec"), namespace)
assert abs(namespace["direct_loss"] - 0.01) < 1e-12
assert abs(namespace["tool_loss"] - 0.12) < 1e-12
assert namespace["tool_loss"] >= namespace["direct_loss"]

L, c, pt = Decimal("10"), Decimal("0.02"), Decimal("0.99")
cases = []
for pd in map(Decimal, ["0.90", "0.999", "0.988"]):
    direct, tool = (1-pd)*L, (1-pt)*L+c
    cases.append({"direct_probability": str(pd), "direct_loss": str(direct),
                  "tool_loss": str(tool), "selected": "TOOL" if tool < direct else "DIRECT"})
assert [r["selected"] for r in cases] == ["TOOL", "DIRECT", "DIRECT"]
assert pt-c/L == Decimal("0.988")
assert Decimal("0.999")*Decimal("0.1") == Decimal("0.0999")

# Load only original method functions, never training or result commentary constants.
method_namespace = {"json": json, "math": math, "copy": copy, "time": time}
locators = []
for path, names in [(retrieval_path, {"call_tool"}),
                    (applications_path, {"_parse_json_action", "_tool_episode"})]:
    tree = ast.parse(path.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    for n in nodes:
        locators.append({"path": str(path.relative_to(ROOT)), "function": n.name,
                         "line_start": n.lineno, "line_end": n.end_lineno})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), method_namespace)

replayed = []
for index, sample in enumerate(data["results"]["samples"]):
    match = re.fullmatch(r"CALC:(add|multiply)\((\d+),(\d+)\)", sample["question"])
    record = {"operation": sample["operation"], "family": sample["family"],
              "answer": sample["expected"], "messages": copy.deepcopy(sample["trace"][0]["generation"]["messages"][:2])}
    if match:
        record.update(a=int(match[2]), b=int(match[3]))
        assert sample["expected"] == (record["a"]+record["b"] if match[1] == "add" else record["a"]*record["b"])
    else:
        match_copy = re.fullmatch(r"COPY:(\d+)", sample["question"])
        assert match_copy and sample["expected"] == int(match_copy[1])
    events = [event["generation"] for event in sample["trace"] if "generation" in event]
    cursor = [0]

    def replay_sample(model, messages, ctx, tokens=64):
        generation = copy.deepcopy(events[cursor[0]])
        if cursor[0] > 0:
            assert generation["messages"] == messages
        generation["messages"] = copy.deepcopy(messages)
        cursor[0] += 1
        return generation

    method_namespace["_sample"] = replay_sample
    result = method_namespace["_tool_episode"](None, record, None, max_steps=sample["max_steps"])
    assert cursor[0] == len(events)
    for field in ["status", "answer", "answer_correct", "required_tool_behavior", "correct", "actual_tool_calls", "generated_tokens"]:
        assert result[field] == sample[field], (index, field, result[field], sample[field])
    for old, new in zip(sample["trace"], result["trace"], strict=True):
        for field in ["executed", "parsed", "finite_result", "tool_result", "request_matches_question", "error"]:
            assert old.get(field) == new.get(field), (index, field)
    replayed.append(result)

failed = replayed[2]
assert failed["question"] == "CALC:multiply(9,9)"
assert failed["trace"][0]["tool_result"] == 81
assert failed["trace"][1]["generation"]["samples"][0]["generated"] == '{"done":true,"answer":8}}'
assert failed["status"] == "invalid_request" and failed["answer"] is None
assert not failed["correct"]

summary = {
    "environment": {"python": sys.version, "device": "CPU", "model_generation": "none; existing text replay only"},
    "original_input_sha256": {"tools_json": sha(data_path), "applications_py": sha(applications_path), "retrieval_py": sha(retrieval_path)},
    "variant_original_fence_stdout": stream.getvalue(),
    "decimal_cost_cases": cases,
    "break_even_direct_probability": "0.988; strict tool_loss < direct_loss selects DIRECT on tie",
    "sequence_probability_counterexample": "0.999 * 0.1 = 0.0999; this is sequence likelihood, not task correctness",
    "original_methods": locators,
    "raw_pointers_read": ["/revision", "/device", "/seed", "/torch_version", "/python_version", "/code_sha256/scripts/course_experiments/applications.py", "/code_sha256/tiny_perceptron/retrieval.py", "/results/protocol/roles", "/results/protocol/external_result_serialization", "/results/protocol/allowlist", "/results/protocol/final", "/results/protocol/max_steps_including_done", "/results/samples/*/{question,expected,operation,family,status,answer,answer_correct,required_tool_behavior,correct,trace,max_steps,actual_tool_calls,generated_tokens}"],
    "replay_counts": {"normal_episodes": len(replayed), "correct": sum(r["correct"] for r in replayed),
                      "actual_calls": sum(r["actual_tool_calls"] for r in replayed), "statuses": dict(Counter(r["status"] for r in replayed))},
    "failure_pointer": "/results/samples/2",
    "failure_evidence": {"executed_result": 81, "model_text": '{"done":true,"answer":8}}',
                         "parse_error": failed["trace"][1]["error"], "task_correct": failed["correct"]},
    "scope": "Arithmetic is hypothetical. Replay verifies original recorded calculations and criteria, not a retrained or newly evaluated model; 19/20 is contextual and not B.8's 0.99 assumption.",
}
(HERE / "bounded-verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
