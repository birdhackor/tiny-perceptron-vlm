"""Bounded CPU audit of original fence and saved JSON generations; no models."""
import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


fence = HERE / "original-fence/fence-1.py"
namespace = {"__name__": "__main__"}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(fence.read_bytes(), str(fence), "exec"), namespace)
expected_stdout = (
    '{"answer":4} 格式 True 內容 True\n'
    '答案是：{"answer":4} 格式 False 內容 False\n'
    '{"answer":5} 格式 True 內容 False\n'
)
assert stdout.getvalue() == expected_stdout
print("ORIGINAL FENCE")
print(stdout.getvalue(), end="")

# Reuse the exact original for-loop AST with only its answers input replaced.
tree = ast.parse(fence.read_bytes())
loop = next(node for node in tree.body if isinstance(node, ast.For))
bounded = compile(ast.Module(body=[loop], type_ignores=[]), str(fence) + ":bounded-inputs", "exec")
cases = [
    ("extra_field", '{"answer":4,"note":"完成"}', (True, False, False)),
    ("missing_field", '{}', (True, False, False)),
    ("array", '[4]', (True, False, False)),
    ("scalar", '4', (True, False, False)),
    ("surrounding_whitespace", ' \n{"answer":4}\t', (True, True, True)),
    ("trailing_prose", '{"answer":4}完成', (False, False, False)),
    ("broken_syntax", '{"answer":4,}', (False, False, False)),
    ("float_four", '{"answer":4.0}', (True, True, True)),
    ("string_four", '{"answer":"4"}', (True, True, False)),
    ("boolean", '{"answer":true}', (True, True, False)),
    ("null", '{"answer":null}', (True, True, False)),
    ("duplicate_name", '{"answer":5,"answer":4}', (True, True, True)),
    ("nan_extension", '{"answer":NaN}', (True, True, False)),
    ("infinity_extension", '{"answer":Infinity}', (True, True, False)),
]
case_results = []
print("BOUNDED INPUT VARIANTS (same original loop)")
for name, text, expected in cases:
    namespace["answers"] = [text]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(bounded, namespace)
    try:
        parsed = json.loads(text)
        parse_success, parsed_type = True, type(parsed).__name__
        value_type = type(parsed.get("answer")).__name__ if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        parse_success, parsed_type, value_type = False, None, None
    observed = (parse_success, namespace["valid"], namespace["correct"])
    assert observed == expected, (name, expected, observed)
    record = {"case": name, "text": text, "parse_success": parse_success,
              "python_type": parsed_type, "answer_type": value_type,
              "format": namespace["valid"], "content": namespace["correct"]}
    case_results.append(record)
    print(json.dumps(record, ensure_ascii=False, allow_nan=False))

# Read stored experiment bytes and require exact recorded historical source hashes.
result_path = HERE / "inputs/current/docs/course-experiments/results/style.json"
result = json.loads(result_path.read_bytes())
for relative in ("scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py",
                 "tiny_perceptron/data.py", "tiny_perceptron/tokenization.py"):
    assert sha(HERE / "inputs/historical" / relative) == result["code_sha256"][relative]

# Execute only historical _style_metrics; no import of training or model code.
behavior_path = HERE / "inputs/historical/scripts/course_experiments/behavior.py"
behavior_tree = ast.parse(behavior_path.read_bytes())
fn = next(node for node in behavior_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_style_metrics")
rubric_ns = {"json": json}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(behavior_path), "exec"), rubric_ns)

report = result["results"]["after"]["test"]
selected = [sample for sample in report["samples"] if sample.get("style") == "json"]
assert len(selected) == 7
rows, sample_results = [], []
for sample in selected:
    prompt = sample["messages"][0]["content"]
    match = re.fullmatch(r"style=json; (\d+)\+(\d+)=\?", prompt)
    assert match, prompt
    a, b = map(int, match.groups())
    expected = a + b  # Independently derive truth from prompt, not sample labels.
    rows.append({"a": a, "b": b, "style": "json", "family": sample["family"]})
    generated = sample["generated"]
    value = json.loads(generated)
    valid = isinstance(value, dict) and set(value) == {"answer"} and type(value["answer"]) is int
    content = valid and value["answer"] == expected
    assert sample["expected"] == json.dumps({"answer": expected})
    assert sample["style_correct"] == valid and sample["content_correct"] == content
    ids = sample["generated_ids"]
    assert ids.count(2) == 1 and ids[-1] == 2
    assert all(8 <= token < 264 for token in ids[:-1])
    raw_text = bytes(token - 8 for token in ids[:-1]).decode("utf-8")
    assert raw_text == generated  # Only designated EOS removed; no prose cleanup.
    sample_results.append({"prompt": prompt, "generated": generated, "truth": expected,
                           "value": value["answer"], "value_type": type(value["answer"]).__name__,
                           "valid": valid, "content_correct": content, "eos": sample["eos"],
                           "answer_byte_tokens": len(ids) - 1, "generated_tokens_including_eos": len(ids)})
recomputed = rubric_ns["_style_metrics"]({"samples": [dict(sample) for sample in selected]}, rows)["rubric"]["json"]
assert recomputed == {"records": 7, "content_correct": 0, "style_correct": 7, "json_valid": 7}
assert recomputed == report["rubric"]["json"]
assert sum(row["valid"] for row in sample_results) == 7
assert sum(row["content_correct"] for row in sample_results) == 0
assert len(report["samples"]) == report["records"] == 27
print("SAVED EXPERIMENT RECOMPUTATION")
for sample in sample_results:
    print(json.dumps(sample, ensure_ascii=False, allow_nan=False))
print("historical_rubric", json.dumps(recomputed, ensure_ascii=False))
summary = {
    "device": "cpu; stdlib only; no training, model load or generation",
    "python": platform.python_version(), "python_executable": sys.executable,
    "original_fence_sha256": sha(fence), "original_fence_output_exact": True,
    "bounded_cases": case_results,
    "saved_result_sha256": sha(result_path), "saved_run_revision": result["revision"],
    "saved_run_device": result["device"], "saved_run_python": result["python_version"],
    "saved_run_torch": result["torch_version"],
    "json_test_samples": sample_results, "rubric": recomputed,
    "denominators": {"json_addition_records": 7, "all_conditional_test_records": 27,
                     "distinct_json_addition_families": len({row["family"] for row in rows}),
                     "training_steps_recorded": result["results"]["training"]["steps"],
                     "generation_budget_tokens_recorded": 96},
    "tolerance": "Exact bool/int/string and SHA equality; no rounding or stochastic run.",
    "scope": "Recomputes saved outputs and executes historical scoring; does not retrain or rerun a model.",
}
(HERE / "verification-results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print("ALL ASSERTIONS PASSED")
