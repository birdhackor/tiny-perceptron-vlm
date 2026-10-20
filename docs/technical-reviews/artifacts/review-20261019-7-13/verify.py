"""核對示例變體、固定配方與已保存實驗；不重跑訓練。"""

import ast
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import split_records
from scripts.course_experiments.run import experiment_spec, list_assets
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_code(code):
    namespace = {}
    output = io.StringIO()
    error = None
    with contextlib.redirect_stdout(output):
        try:
            exec(compile(code, "frozen-example", "exec"), namespace)
        except Exception as exception:
            error = type(exception).__name__
    return {"stdout": output.getvalue(), "exception": error, "scores": namespace.get("scores")}


def command(argv):
    completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return {"command_argv": argv, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/7.13.md"
source = frozen.read_text()
code = re.search(r"```python\n(.*?)```", source, re.S)[1]
correct = code.replace("3}", "2}")
fixed_assertion = correct.replace('and not scores["1+1內容正確"]', 'and scores["1+1內容正確"]')
renamed = correct.replace('reply = \'{"answer": 2}\'', 'reply = \'{"result": 2}\'')
variants = {
    "original": run_code(code),
    "answer_two_original_assertion": run_code(correct),
    "answer_two_both_true_assertion": run_code(fixed_assertion),
    "renamed_result": run_code(renamed),
}
assert variants["original"]["scores"] == {"格式符合": True, "1+1內容正確": False}
assert variants["original"]["exception"] is None
assert variants["answer_two_original_assertion"]["scores"] == {"格式符合": True, "1+1內容正確": True}
assert variants["answer_two_original_assertion"]["exception"] == "AssertionError"
assert variants["answer_two_both_true_assertion"]["exception"] is None
assert variants["renamed_result"]["scores"] == {"格式符合": False, "1+1內容正確": False}

boundaries = []
for reply in ['{"answer": true}', '{"answer": false}', '{"answer": 2.0}', '{}', '{"answer": 2, "extra": 0}', '[]', 'not json']:
    boundaries.append({"reply": reply, **run_code(code.replace("'{\"answer\": 3}'", repr(reply)))})

python_semantics = {
    "true_type": type(json.loads("true")).__name__,
    "false_type": type(json.loads("false")).__name__,
    "bool_subclass_int": issubclass(bool, int),
    "isinstance_true_int": isinstance(True, int),
    "true_equals_one": True == 1,
    "true_exact_int": type(True) is int,
    "integer_exact_int": type(2) is int,
    "missing_dict_get": {}.get("answer"),
    "dict_keys": sorted(set({"answer": 3})),
    "one_plus_one": 1 + 1,
}
assert all(python_semantics[key] for key in ("bool_subclass_int", "isinstance_true_int", "true_equals_one", "integer_exact_int"))
assert python_semantics["true_exact_int"] is False

result_path = ROOT / "docs/course-experiments/results/sft.json"
record = json.loads(result_path.read_text())
results = record["results"]
test = results["after"]["test"]
tok = ByteTokenizer()
parts = split_records(generate_records("attributes-sft"), seed=record["seed"])
sample_checks = []
for row, sample in zip(parts["test"], test["samples"], strict=True):
    ids = sample["generated_ids"]
    raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
    exact = raw == tok.encode(sample["expected"])
    ended = tok.eos_id in ids
    check = {
        "question": sample["messages"],
        "expected": sample["expected"],
        "generated_ids": ids,
        "decoded": tok.decode(raw),
        "recomputed_exact": exact,
        "recorded_exact": sample["exact"],
        "recomputed_eos": ended,
        "recorded_eos": sample["eos"],
        "matches_regenerated_question": row["messages"][:-1] == sample["messages"],
        "matches_regenerated_answer": row["messages"][-1]["content"] == sample["expected"],
    }
    assert check["matches_regenerated_question"] and check["matches_regenerated_answer"]
    assert exact is sample["exact"] and ended is sample["eos"]
    sample_checks.append(check)
matches = sum(sample["recomputed_exact"] for sample in sample_checks)
assert matches == test["matches"] == 5
assert len(sample_checks) == test["records"] == 10
assert matches / len(sample_checks) == test["exact_match"] == 0.5
assert results["training"]["steps"] == 900
assert results["checkpoint"] == "model.pt"
assert round(record["elapsed_seconds"], 1) == 31.7

code_paths = [
    "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
    "scripts/course_experiments/run.py", "scripts/fetch_training_assets.py",
    "tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/training.py",
    "scripts/prepare_data.py", "docs/course-experiments/plan.json",
    "assets/training/manifest.json", "course/training.md",
    "docs/course-experiments/results/sft.json", "docs/course-experiments/results/sft_ablation.json",
]
file_checks = []
for path in code_paths:
    actual = sha(ROOT / path)
    historical = record.get("code_sha256", {}).get(path)
    file_checks.append({"path": path, "sha256": actual, "recorded_code_sha256": historical,
                        "matches_run_code": actual == historical if historical else None})

selected = {
    "scripts/course_experiments/text.py": {"run_sft", "run_sft_ablation", "_evaluations", "_natural_chat_pilot", "_asset_rows"},
    "scripts/course_experiments/common.py": {"new_lm", "evaluate_lm", "dependency"},
    "scripts/course_experiments/run.py": {"main"},
    "tiny_perceptron/data.py": {"ByteTokenizer"},
    "tiny_perceptron/model.py": {"generate"},
}
code_inspection = []
for path, names in selected.items():
    body = (ROOT / path).read_text()
    tree = ast.parse(body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            code_inspection.append({"path": path, "identifier": node.name,
                                    "source": ast.get_source_segment(body, node)})
runner = (ROOT / "scripts/course_experiments/run.py").read_text().splitlines()
timing_lines = [line for line in runner if any(token in line for token in ("started =", "elapsed =", "synchronize", "experiment(context)", "timing_scope", "training, evaluation"))]

ablation_path = ROOT / "docs/course-experiments/results/sft_ablation.json"
ablation = json.loads(ablation_path.read_text())
ablation_before = ablation["results"]["before"]["A_attributes"]["test"]
assert ablation_before == test

commands = [
    command([str(ROOT / ".venv/bin/python"), "scripts/fetch_training_assets.py", "--help"]),
    command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"]),
    command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--list-assets", "sft"]),
]
assert all(item["exit_code"] == 0 for item in commands)
spec = experiment_spec("sft")
assert spec["module"] == "text" and spec["function"] == "run_sft"
assert spec["assets"] == ["ultrachat-sft"]

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-13/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "source_sha256": sha(frozen), "python_semantics": python_semantics,
    "example_variants": variants, "boundaries": boundaries,
    "aggregation_derivation": {"before": [0.0, 1.0], "after": [1.0, 0.0], "before_sum": 1.0, "after_sum": 1.0,
                               "details": "等權加總使格式增加的一分恰好抵消內容減少的一分；兩個向量仍保留不同座標。"},
    "run_record": {
        "path": str(result_path.relative_to(ROOT)), "sha256": sha(result_path),
        "metadata": {key: record[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status")},
        "training": {key: results["training"][key] for key in ("steps", "records", "effective_tokens", "checkpoint", "seconds")},
        "dataset": results["data"], "test_recomputation": {"records": len(sample_checks), "matches": matches, "exact_match": matches / len(sample_checks), "eos_count": sum(s["recomputed_eos"] for s in sample_checks)},
        "samples": sample_checks,
        "pretrain_sft_test": {key: results["pretrain_then_sft"]["after_sft"]["test"][key] for key in ("matches", "records", "exact_match")},
        "elapsed_rounded_to_tenth": round(record["elapsed_seconds"], 1),
        "checkpoint_receipts": [a for a in record["artifacts"] if a["path"] in {"model.pt", "pretrain-sft.pt", "start.pt", "dataset.json"}],
    },
    "regenerated_split_counts": {name: len(rows) for name, rows in parts.items()},
    "ablation_before_equals_sft_after": ablation_before == test,
    "experiment_spec": spec, "listed_assets": list_assets("sft"),
    "file_checks": file_checks, "code_inspection": code_inspection, "timing_code": timing_lines,
    "commands": commands,
    "historical_runner_lookup": command(["git", "show", record["revision"] + ":scripts/course_experiments/run.py"]),
    "scope": "實跑示例、練習、CLI 解析與既有紀錄的重算；沒有訓練模型、下載權重、重新測量 GPU 時間。run.py 與紀錄中的程式雜湊不同；計時範圍依紀錄的 timing_scope 與現行 execute 的計時區間交叉核對，不宣稱取回舊版 runner。",
    "result": "所有示例、樣本、分母、步數、四捨五入與 CLI 核對通過。",
}
with Path(__file__).with_name("verification.json").open("x", encoding="utf-8") as target:
    json.dump(output, target, ensure_ascii=False, indent=2)
    target.write("\n")
print(json.dumps({"result": output["result"], "test": output["run_record"]["test_recomputation"], "elapsed": output["run_record"]["elapsed_rounded_to_tenth"], "variants": variants}, ensure_ascii=False))
