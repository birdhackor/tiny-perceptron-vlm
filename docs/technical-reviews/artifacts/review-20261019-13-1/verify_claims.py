import ast
import contextlib
import hashlib
import inspect
import io
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments import behavior
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import arithmetic_records


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def serialized(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


dpo_path = "docs/course-experiments/results/dpo.json"
style_path = "docs/course-experiments/results/style.json"
dpo = json.loads((ROOT / dpo_path).read_bytes())
style = json.loads((ROOT / style_path).read_bytes())
require(dpo["seed"] == style["seed"] == 42, "兩份紀錄的種子不同")
records = arithmetic_records()
arithmetic = split_records(records, seed=dpo["seed"])
pairs = behavior._preference_parts(arithmetic)
require(len(records) == 8 * 8 == 64, "題數不符")
require(len({row["family"] for row in records}) == 36, "家族總數不符")
require({(row["a"], row["b"]) for row in records} == {(a, b) for a in range(8) for b in range(8)}, "題目不是完整笛卡兒積")

split_results = {}
for name, rows in arithmetic.items():
    families = {row["family"] for row in rows}
    expected = dpo["results"]["data"][name]
    require(len(rows) == expected["records"], "題目對數與紀錄不同")
    require(len(families) == expected["families"], "家族數與紀錄不同")
    require(sha(serialized(pairs[name])) == expected["sha256"], "偏好資料的逐位元組雜湊不同")
    require(sha(serialized(rows)) == style["results"]["arithmetic_data"][name]["sha256"], "加法底座資料的逐位元組雜湊不同")
    for row, pair in zip(rows, pairs[name], strict=True):
        require(pair["chosen"] == str(row["a"] + row["b"]), "較佳答案不是正確的和")
        require(pair["rejected"] == str(row["a"] + row["b"] + 1), "較差答案不符生成規則")
    formatted = [{**row, "rejected": row["chosen"] + "; answer complete"} for row in pairs[name]]
    require(sha(serialized(formatted)) == dpo["results"]["format_only"]["data"][name]["sha256"], "格式支線資料不同")
    split_results[name] = {
        "records": len(rows),
        "families": len(families),
        "family_names": sorted(families),
        "pair_data_sha256": sha(serialized(pairs[name])),
        "base_data_sha256": sha(serialized(rows)),
        "format_data_sha256": sha(serialized(formatted)),
    }
for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    require(not ({row["family"] for row in arithmetic[first]} & {row["family"] for row in arithmetic[second]}), "家族跨組")
for row in records:
    require(row["family"] == f'{min(row["a"], row["b"])}+{max(row["a"], row["b"])}', "交換加數的家族鍵不同")

frozen_path = "docs/course-review-20261019/freeze-01/sources/13.1.md"
frozen = (ROOT / frozen_path).read_bytes()
fences = re.findall(rb"```python\n(.*?)```", frozen, re.S)
require(len(fences) == 1, "頁面程式區塊數量不同")
page_runs = []
for code in [fences[0], fences[0].replace("只回數字：".encode(), b"")]:
    namespace = {}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, frozen_path, "exec"), namespace)
    require(namespace["chosen"] == "A", "固定選擇意外改變")
    page_runs.append({"prompt": namespace["pair"]["prompt"], "chosen": namespace["chosen"], "stdout": stream.getvalue()})


class Model:
    def eval(self):
        return self

    def requires_grad_(self, enabled):
        return self

    def state_dict(self):
        return {}


actual_functions = {name: getattr(behavior, name) for name in ["load_lm", "_preference_evaluate", "_dpo_train", "_evaluations", "_save_splits", "_natural_dpo_pilot"]}
train_signature = inspect.signature(actual_functions["_dpo_train"])


def route_check(metric):
    events = []

    def evaluate(model, reference, rows):
        events.append({"operation": "evaluate", "records": len(rows), "families": sorted({r["family"] for r in rows})})
        return {"metric": metric}

    def train(model, reference, rows, ctx, name, beta=0.1, count=250):
        require({r["family"] for r in rows} == {r["family"] for r in pairs["train"]}, "訓練函式收到非訓練家族")
        events.append({"operation": "train", "name": name, "beta": beta, "steps": count, "records": len(rows)})
        return {"steps": count}

    behavior.load_lm = lambda path, device: Model()
    behavior._preference_evaluate = evaluate
    behavior._dpo_train = train
    behavior._evaluations = lambda model, parts: {key: len(value) for key, value in parts.items()}
    behavior._save_splits = lambda ctx, parts, name="data": {key: len(value) for key, value in parts.items()}
    behavior._natural_dpo_pilot = lambda ctx: {"scope": "未執行自然語言支線"}
    try:
        ctx = SimpleNamespace(seed=42, device="cpu", step_scale=1.0, dependency=lambda name, artifact: f"{name}/{artifact}")
        behavior.run_dpo(ctx)
    finally:
        for name, function in actual_functions.items():
            setattr(behavior, name, function)
    return events


trace = route_check(-1000.0)
require(trace == route_check(1000.0), "評估分數改變了訓練流程")
training = [event for event in trace if event["operation"] == "train"]
require([(event["name"], event["beta"], event["steps"]) for event in training] == [("model", 0.1, 250), ("beta1", 1.0, 250), ("format-model", 0.1, 200)], "設定與預期不同")
require(train_signature.parameters["count"].default == 250, "實際訓練函式預設步數不同")
for name in ["model", "beta1"]:
    require(dpo["results"]["runs"][name]["training"]["steps"] == 250, "紀錄中的更新步數不同")
    require(dpo["results"]["runs"][name]["beta"] == next(event["beta"] for event in training if event["name"] == name), "紀錄中的 beta 不同")
    for split in ["validation", "test"]:
        require(dpo["results"]["runs"][name]["preference"][split]["records"] == len(pairs[split]), "訓練後評估題數不符")
for split in ["validation", "test"]:
    require(dpo["results"]["before"][split]["records"] == len(pairs[split]), "底座評估題數不符")

behavior_path = "scripts/course_experiments/behavior.py"
target_sha = dpo["code_sha256"][behavior_path]
commits = subprocess.check_output(["git", "log", "--all", "--format=%H", "--", behavior_path], text=True).splitlines()
matched = None
for commit in commits:
    candidate = subprocess.run(["git", "show", commit + ":" + behavior_path], capture_output=True)
    if candidate.returncode == 0 and sha(candidate.stdout) == target_sha:
        historical = ast.parse(candidate.stdout)
        current = ast.parse((ROOT / behavior_path).read_bytes())
        functions_equal = {}
        for name in ["_preference_parts", "_dpo_train", "_preference_evaluate", "run_dpo", "run_style"]:
            old = next(node for node in historical.body if isinstance(node, ast.FunctionDef) and node.name == name)
            new = next(node for node in current.body if isinstance(node, ast.FunctionDef) and node.name == name)
            functions_equal[name] = ast.dump(old) == ast.dump(new)
        require(all(functions_equal.values()), "歷史實驗的相關函式與現行版本不同")
        matched = {"commit": commit, "sha256": target_sha, "functions_ast_equal": functions_equal}
        break
require(matched is not None, "找不到與實驗來源雜湊相符的程式")

paths = [dpo_path, style_path, frozen_path, behavior_path, "scripts/course_experiments/text.py", "scripts/course_experiments/common.py"]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-1/verify_claims.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "input_sha256": {path: sha((ROOT / path).read_bytes()) for path in paths},
    "numeric_examples": {"addition": 2 + 2, "two_pairs_of_chopsticks": 2 * 2, "incorrect_candidate": 5, "total_questions": len(records), "total_families": 36},
    "splits": split_results,
    "all_pair_labels_correct": True,
    "family_sets_disjoint": True,
    "base_and_dpo_splits_match_byte_hashes": True,
    "format_branch_labels_correct": True,
    "page_runs": page_runs,
    "control_flow_trace": trace,
    "extreme_metric_values_same_training_route": True,
    "recorded_experiment": {"seed": dpo["seed"], "device": dpo["device"], "python": dpo["python_version"], "torch": dpo["torch_version"], "step_scale": dpo["step_scale"], "training_steps": {name: dpo["results"]["runs"][name]["training"]["steps"] for name in ["model", "beta1"]}},
    "historical_behavior_source": matched,
    "scope": "執行真正的題目與候選生成、家族切分與頁面程式；訓練和評估以替身追蹤呼叫順序，沒有執行參數更新或重測 CUDA 品質。來源函式另以 AST 與歷史實驗雜湊核對。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
