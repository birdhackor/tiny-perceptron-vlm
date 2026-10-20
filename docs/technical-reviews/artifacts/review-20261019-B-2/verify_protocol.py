"""核對工具呼叫、非有限值處理與保存的實驗紀錄；不訓練模型。"""

import ast
import hashlib
import inspect
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import applications as app
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.retrieval import call_tool


def reject(function, value):
    try:
        function(value)
    except (ValueError, TypeError, KeyError) as error:
        return {"rejected": True, "type": type(error).__name__, "error": str(error)}
    raise AssertionError(f"應被拒絕的輸入卻通過：{value!r}")


report = {
    "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"},
    "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("tiny_perceptron/retrieval.py", "tiny_perceptron/data.py", "scripts/course_experiments/applications.py", "tests/test_application_protocols.py", "docs/course-experiments/results/tools.json")},
}
frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/B.2.md").read_text()
code = frozen.split("```python\n")[2].split("```")[0]
frozen_function = ast.parse(code).body[0]
current_function = ast.parse(inspect.getsource(call_tool)).body[0]
assert ast.dump(frozen_function) == ast.dump(current_function)
report["excerpt_matches_repository_function"] = True

request = {"name": "add", "arguments": {"a": 2, "b": 3}}
assert call_tool(request) == call_tool(json.dumps(request)) == 5
assert call_tool({"name": "multiply", "arguments": {"a": 2, "b": 3}}) == 6
report["valid_requests"] = {"add": 5, "multiply": 6, "dict_and_json_agree": True}
invalid = {
    "unknown_tool": {"name": "delete_all", "arguments": {"a": 2, "b": 3}},
    "extra_field": request | {"extra": 0},
    "missing_field": {"name": "add"},
    "arguments_not_object": {"name": "add", "arguments": [2, 3]},
    "extra_argument": {"name": "add", "arguments": {"a": 2, "b": 3, "c": 4}},
    "missing_argument": {"name": "add", "arguments": {"a": 2}},
    "bool": {"name": "add", "arguments": {"a": True, "b": 3}},
    "string": {"name": "add", "arguments": {"a": "2", "b": 3}},
}
report["schema_rejections"] = {key: reject(call_tool, value) for key, value in invalid.items()}
assert set(request) == {"name", "arguments"}
assert list(request["arguments"].values()) == [2, 3]
assert any(type(v) not in (int, float) for v in (True, 3))
assert not any(type(v) not in (int, float) for v in (2.5, 3))
assert isinstance("text", str) and not isinstance(request, str)
assert type(json.loads("true")) is bool and isinstance(True, int) and True + 0 == 1
assert type(True) not in (int, float)
assert type(json.loads('"2"')) is str
assert call_tool({"name": "add", "arguments": {"a": 2.5, "b": 3}}) == 5.5
assert call_tool({"name": "add", "arguments": {"a": 2, "b": 4}}) == 6
report["python_semantics"] = {"true_type": "bool", "true_isinstance_int": True, "true_exact_numeric_type": False, "string_type": "str", "float_type": type(json.loads("2.5")).__name__, "float_exercise_result": 5.5, "wrong_parameters_valid_result": 6}
assert math.isfinite(1e308) and math.isinf(1e999) and math.isinf(1e308 * 1e308)
assert not math.isfinite(float("nan")) and not math.isfinite(float("-inf"))
assert math.isinf(call_tool('{"name":"multiply","arguments":{"a":1e308,"b":1e308}}'))
assert math.isinf(call_tool('{"name":"add","arguments":{"a":1e999,"b":3}}'))
assert math.isnan(call_tool('{"name":"add","arguments":{"a":NaN,"b":3}}'))
assert call_tool('{"name":"add","name":"multiply","arguments":{"a":2,"b":3}}') == 6
report["nonfinite"] = {"finite_input": "1e308", "literal_overflow": repr(1e999), "product_overflow": repr(1e308 * 1e308), "plain_call_accepts_nonfinite": True, "plain_call_accepts_duplicate_name": True, "float_max": repr(sys.float_info.max)}

strict_inputs = ["NaN", "Infinity", "-Infinity", "1e999", "-1e999", "[]", '{"name":"add","name":"multiply","arguments":{"a":2,"b":3}}', '{"name":"add","arguments":{"a":2,"a":3,"b":3}}', r'{"name":"add","arguments":{"a":2,"\u0061":3,"b":3}}', '{"x":{"deep":{"a":1,"a":2}}}']
report["strict_rejections"] = [{"input": value, **reject(app._parse_json_action, value)} for value in strict_inputs]

record = {"family": "pair:2:3", "operation": "multiply", "a": 2, "b": 3, "answer": 6, "messages": [{"role": "system", "content": app._TOOLS_SYSTEM}, {"role": "user", "content": "CALC:multiply(2,3)"}]}
original_sample, original_parser = app._sample, app._parse_json_action


def episode_for(text, special=()):
    tokenizer = ByteTokenizer()
    def sampled(*args, **kwargs):
        return {"samples": [{"generated": text, "invalid_special_tokens": list(special), "eos": True}], "generated_tokens": len(tokenizer.encode(text)) + len(special) + 1}
    app._sample = sampled
    try:
        return app._tool_episode(None, record, SimpleNamespace(device="cpu"))
    finally:
        app._sample = original_sample


overflow = episode_for('{"name":"multiply","arguments":{"a":1e308,"b":1e308}}')
assert overflow["status"] == "invalid_tool_result" and overflow["actual_tool_calls"] == 1
assert overflow["trace"][0]["executed"] is True and overflow["trace"][0]["finite_result"] is False
assert overflow["trace"][0]["tool_result"] == "inf" and not overflow["correct"]
json.dumps(overflow, allow_nan=False)
report["overflow_episode"] = overflow

def must_not_parse(text):
    raise AssertionError("特殊 token 應在 JSON 解析之前遭拒")

app._parse_json_action = must_not_parse
try:
    special = episode_for('{"name":"multiply","arguments":{"a":2,"b":3}}', [ByteTokenizer.user_id])
finally:
    app._parse_json_action = original_parser
assert special["status"] == "invalid_request" and special["actual_tool_calls"] == 0
report["special_token_episode"] = special
report["tokenizer"] = {"user_id": ByteTokenizer.user_id, "eos_id": ByteTokenizer.eos_id}

saved = json.loads((ROOT / "docs/course-experiments/results/tools.json").read_text())
results = saved["results"]
injected = results["injected_protocol_checks_not_model_scores"]
assert len(injected) == 7 and all(item["rejected"] for item in injected)
def parsed_call(text):
    return call_tool(app._parse_json_action(text))
replay = [{"input": item["input"], **reject(parsed_call, item["input"])} for item in injected]
assert [{k: v for k, v in item.items() if k != "type"} for item in replay] == injected
assert len(results["samples"]) == results["accuracy"]["denominator"] == results["split"]["test"]["records"] == 20
events = [event for row in results["samples"] + results["one_step_cap_samples"] for event in row["trace"] if event.get("executed")]
assert all(event["finite_result"] for event in events)
assert all(row["a"] <= 9 and row["b"] <= 9 for row in app._tool_records() if row["operation"] != "copy")
report["saved_experiment_audit"] = {"device": saved["device"], "revision": saved["revision"], "seed": results["seed"], "test_count": len(results["samples"]), "accuracy_denominator": results["accuracy"]["denominator"], "injected_count": len(injected), "injected": injected, "replayed": replay, "executed_events_including_one_step_cap": len(events), "all_executed_results_finite": True, "arithmetic_operand_range": [0, 9]}
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
