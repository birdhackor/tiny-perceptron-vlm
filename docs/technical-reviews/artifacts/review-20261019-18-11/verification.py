"""重算人工算例、留存逐題證據，並對照資料、控制條件與判分器。"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.behavior import _conversation, _style_metrics, _style_record
from scripts.course_experiments.common import split_records
from scripts.course_experiments.compression import _style_scores
from scripts.course_experiments.text import arithmetic_records
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.tokenization import generation_report


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def run_example(code):
    namespace, stream = {}, io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, "人工回答算例", "exec"), namespace)
    return {
        "stdout": stream.getvalue(),
        "agreement": namespace["agreement"],
        "checks": namespace["checks"],
        "rounded_pass_rate": round(sum(namespace["checks"]) / 3, 4),
        "copy_preserves_values": namespace["teacher_answers"] == namespace["student_answers"],
        "copy_creates_new_list": namespace["teacher_answers"] is not namespace["student_answers"],
    }


freeze_path = "docs/course-review-20261019/freeze-01/sources/18.11.md"
frozen = (ROOT / freeze_path).read_text()
code = re.search(r"```python\n(.*?)```", frozen, re.S).group(1)
baseline = run_example(code)
changed = run_example(code.replace('["不知道", "不知道", "不知道"]', '["2", "不知道", "不知道"]', 1))
assert baseline["agreement"] == changed["agreement"] == 1.0
assert baseline["checks"] == [False, False, True]
assert changed["checks"] == [True, False, True]
assert baseline["rounded_pass_rate"] == 0.3333
assert changed["rounded_pass_rate"] == 0.6667
try:
    run_example(code.replace('["不知道", "不知道", "不知道"]', '[2, "不知道", "不知道"]', 1))
except AttributeError as error:
    integer_failure = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("整數回答應在字串方法失敗")

example = {
    "baseline": baseline,
    "teacher_first_answer_corrected": changed,
    "integer_failure": integer_failure,
    "exact_match_for_unknown": "不知道" == "資訊不足",
    "zip_result": list(zip(["a", "b"], ["c", "d"])),
    "boolean_sum": sum([False, False, True]),
    "arithmetic": {"1+1": 1 + 1, "2+2": 2 + 2},
    "scope": "三題人工文字清單，沒有訓練或生成；None 是題目規則，不是對個人晚餐的實測。",
}

result_path = "docs/course-experiments/results/distillation.json"
distillation = json.loads((ROOT / result_path).read_bytes())
style_result_path = "docs/course-experiments/results/style.json"
style_result = json.loads((ROOT / style_result_path).read_bytes())
style = distillation["results"]["tasks"]["style_transfer"]
arithmetic = split_records(arithmetic_records(), seed=42)
parts = {
    split: [_style_record(row, mode, True) for row in rows for mode in ("concise", "vivid", "json")]
    for split, rows in arithmetic.items()
}
dates = []
for day in range(1, 25):
    date = f"2026-10-{day:02d}"
    dates.append(_conversation(f"task=date;date={date};confirm", "已確認" + date + "。", "date-" + date, style="clarification"))
    dates.append(_conversation(f"task=date;id={day};date=?;confirm", "請提供日期。", "date-" + date, style="clarification"))
date_parts = split_records(dates, seed=42)
parts = {split: rows + date_parts[split] for split, rows in parts.items()}
dataset_digest = hashlib.sha256((json.dumps(parts, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
assert dataset_digest == style["data"]["sha256"]
assert {k: len(v) for k, v in parts.items()} == style["data"]["counts"]
families = {k: set(row["family"] for row in rows) for k, rows in parts.items()}
assert not families["train"] & families["test"]
assert not families["validation"] & families["test"]
assert style["teacher_provenance"]["metadata"]["records_sha256"] == style_result["results"]["training"]["records_sha256"]

tok = ByteTokenizer()
branches = {"teacher": (style["teacher_test"], style["teacher_style"])}
methods = ["w32_ce", "w32_teacher_hard", "w32_ce_kl"]
for method in methods:
    branches[method] = (style["runs"][method]["test"], style["runs"][method]["style"])
summary = {}
for name, (evaluation, recorded_style) in branches.items():
    recalculated = _style_scores(copy.deepcopy(evaluation), parts["test"])
    for key in ("content_correct", "examples", "json_valid", "json_examples", "rubric"):
        assert recalculated[key] == recorded_style[key]
    independent = {}
    row_receipts = []
    for row, sample in zip(parts["test"], evaluation["generated_samples"], strict=True):
        ids = sample["generated_ids"]
        raw = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        valid = all(value >= 8 for value in raw)
        text = bytes(value - 8 for value in raw if value >= 8).decode("utf-8")
        assert text == sample["generated"]
        assert sample["expected"] == row["messages"][-1]["content"]
        assert sample["question"] == row["messages"][0]["content"]
        mode = row["style"]
        if mode == "json":
            parsed = json.loads(text)
            format_ok = type(parsed) is dict and set(parsed) == {"answer"} and type(parsed["answer"]) is int and valid
            content_ok = format_ok and parsed["answer"] == row["a"] + row["b"]
        elif mode == "vivid":
            format_ok = "像把兩組積木合在一起再數" in text and valid
            content_ok = text.split("，", 1)[0].strip() == str(row["a"] + row["b"]) and valid
        elif mode == "concise":
            format_ok = text.strip().isdigit() and valid
            content_ok = text.strip() == str(row["a"] + row["b"]) and valid
        else:
            format_ok = content_ok = raw == tok.encode(sample["expected"]) and valid
        group = independent.setdefault(mode, {"records": 0, "content_correct": 0, "style_correct": 0, "json_valid": 0})
        group["records"] += 1
        group["content_correct"] += int(content_ok)
        group["style_correct"] += int(format_ok)
        group["json_valid"] += int(mode == "json" and format_ok)
        row_receipts.append({"family": row["family"], "question": sample["question"], "expected": sample["expected"], "generated": text, "generated_ids": ids, "content_correct": content_ok, "format_correct": format_ok, "eos": tok.eos_id in ids})
    assert independent == recorded_style["rubric"]
    assert evaluation["answer_nll"] == evaluation["nll_sum"] / evaluation["supervised_tokens"]
    assert evaluation["eos_count"] == sum(r["eos"] for r in row_receipts)
    summary[name] = {"rubric": independent, "samples": row_receipts, "answer_nll": evaluation["answer_nll"], "supervised_tokens": evaluation["supervised_tokens"], "eos_count": evaluation["eos_count"]}

training = {name: style["runs"][name]["training"] for name in methods}
for field in ("initialization_sha256", "batch_plan_sha256", "steps", "optimizer_updates", "batch_size", "training_examples", "effective_supervised_tokens"):
    assert len({run[field] for run in training.values()}) == 1
plan_rng = random.Random(42)
plan = [[plan_rng.randrange(185) for _ in range(16)] for _ in range(300)]
assert hashlib.sha256(json.dumps(plan).encode()).hexdigest() == training["w32_ce"]["batch_plan_sha256"]

def probe_rubric(text, mode, answer, expected, invalid_id=None):
    row = _conversation("測試提示", expected, "測試家族", style=mode)
    if mode != "clarification":
        row.update(a=2, b=2)
    ids = tok.encode(text)
    if invalid_id is not None:
        ids = [invalid_id] + ids
    ids += [tok.eos_id]
    raw = ids[:-1]
    sample = {"generated": text, "generated_ids": ids, "exact": raw == tok.encode(expected)}
    report = _style_metrics({"samples": [sample]}, [row])
    return {"input": text, "kind": mode, "rubric": report["rubric"], "generation": generation_report(tok, ids)}

rubric_probes = [
    probe_rubric('{"answer": 3}', "json", 3, '{"answer": 4}'),
    probe_rubric('{"answer": 4}', "json", 4, '{"answer": 4}'),
    probe_rubric('{"answer": true}', "json", True, '{"answer": 4}'),
    probe_rubric('{"answer": 4}', "json", 4, '{"answer": 4}', tok.user_id),
    probe_rubric("已確認2026-10-02。", "clarification", None, "已確認2026-10-02。"),
    probe_rubric("已確認日期。", "clarification", None, "已確認2026-10-02。"),
]
assert [x["rubric"]["json"]["style_correct"] for x in rubric_probes[:4]] == [1, 1, 0, 0]
assert [x["rubric"]["json"]["content_correct"] for x in rubric_probes[:4]] == [0, 1, 0, 0]
assert [x["rubric"]["clarification"]["content_correct"] for x in rubric_probes[4:]] == [1, 0]

code_paths = ["scripts/course_experiments/compression.py", "scripts/course_experiments/behavior.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "tiny_perceptron/data.py", "tiny_perceptron/tokenization.py"]
code_receipts = {p: {"current_sha256": sha(p), "recorded_sha256": distillation["code_sha256"][p]} for p in code_paths}
historical_revision = "dc8d1ac3cec62d33ad973e54ba7eaf44732d8c17"
historical_command = ["git", "show", historical_revision + ":scripts/course_experiments/compression.py"]
historical_raw = subprocess.check_output(historical_command, cwd=ROOT)
assert hashlib.sha256(historical_raw).hexdigest() == distillation["code_sha256"][code_paths[0]]
old_tree, new_tree = ast.parse(historical_raw), ast.parse((ROOT / code_paths[0]).read_bytes())
ast_comparison = {}
for name in ("_teacher", "_dataset", "_evaluate", "_style_scores", "_fit_text", "_hard_targets", "_distill_case", "run_distillation"):
    old = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    new = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ast_comparison[name] = ast.dump(old) == ast.dump(new)
assert all(ast_comparison.values())

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-18-11/verification.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "inputs": {p: sha(p) for p in [freeze_path, result_path, style_result_path] + code_paths},
    "examples": example,
    "experiment_environment": {k: distillation[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "evidence_status")},
    "data": {"reconstructed_sha256": dataset_digest, "recorded": style["data"], "test_arithmetic_unique_ordered_questions": len({r["messages"][0]["content"].split("; ")[-1] for r in parts["test"] if "a" in r}), "test_arithmetic_families": sorted({r["family"] for r in parts["test"] if "a" in r})},
    "training": training,
    "teacher_provenance": style["teacher_provenance"],
    "hard_target_summary": {k: v for k, v in style["hard_target_generation"].items() if k != "audit"},
    "branches": summary,
    "rubric_probes": rubric_probes,
    "code_receipts": code_receipts,
    "historical_code_check": {"command_argv": historical_command, "sha256": hashlib.sha256(historical_raw).hexdigest(), "same_function_ast": ast_comparison},
    "scope": "實跑人工算例、資料重建及判分；從已存生成 IDs 獨立重算原實驗結果，沒有重新生成、重訓或量測 GPU。只支持這批模板、這個 seed 與記錄的配置。",
    "status": "passed",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
