"""核對固定題目、逐題成績與實驗紀錄；不訓練或下載權重。"""

import contextlib
import hashlib
import io
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

from scripts.course_experiments.common import split_records, text_examples
from scripts.course_experiments.text import arithmetic_records
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_bytes())


record_path = "docs/course-experiments/results/sft_ablation.json"
baseline_path = "docs/course-experiments/results/sft.json"
record = load(record_path)
baseline = load(baseline_path)
results = record["results"]
after = results["runs"]["b-only"]
source_path = "docs/course-review-20261019/freeze-01/sources/7.15.md"
source = (ROOT / source_path).read_bytes()
assert digest(source) == "4dd3a12a31ed31266f763dde0e42c1a4bfe703abfe054fcc0bb876a2647b4fc6", "凍結原文須相符"
code = re.search(rb"```python\n(.*?)```", source, re.S)[1].decode()
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, source_path, "exec"), namespace)
assert len(namespace["a_questions"]) == len(namespace["b_questions"]) == 2, "固定題數須相符"
assert all(value is None for row in namespace["measurements"].values() for value in row.values()), "表格尚無測量值"
namespace["a_questions"].append("紅色方塊是什麼形狀？")
assert len(namespace["a_questions"]) == 3, "練習須增加一題"
assert all(value is None for row in namespace["measurements"].values() for value in row.values()), "新增題目不會填入成績"

parts = {
    "attributes": split_records(generate_records("attributes-sft"), seed=record["seed"]),
    "arithmetic": split_records(arithmetic_records(), seed=record["seed"]),
}
data = {}
for task, splits in parts.items():
    families = {split: {row["family"] for row in rows} for split, rows in splits.items()}
    assert not (families["train"] & families["validation"]), "訓練與驗證家族不能重疊"
    assert not (families["train"] & families["test"]), "訓練與測試家族不能重疊"
    assert not (families["validation"] & families["test"]), "驗證與測試家族不能重疊"
    data[task] = {}
    for split, rows in splits.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        observed = {"records": len(rows), "families": len(families[split]), "sha256": digest(raw)}
        expected = results["data"][task][split]
        assert all(observed[key] == expected[key] for key in observed), "重建資料須符合原紀錄"
        data[task][split] = observed
all_arithmetic = arithmetic_records()
assert len(all_arithmetic) == 64, "加法題須包含全部有序加數對"
assert min(row["a"] for row in all_arithmetic) == min(row["b"] for row in all_arithmetic) == 0
assert max(row["a"] for row in all_arithmetic) == max(row["b"] for row in all_arithmetic) == 7
assert all(row["family"] == f"{min(row['a'], row['b'])}+{max(row['a'], row['b'])}" for row in all_arithmetic)
assert len({row["family"] for row in all_arithmetic}) == 36

tok = ByteTokenizer()
evaluations = {}
for stage, stage_results in (("before", results["before"]), ("b-only", after)):
    evaluations[stage] = {}
    for task, field in (("attributes", "A_attributes"), ("arithmetic", "B_arithmetic")):
        evaluations[stage][field] = {}
        for split in ("validation", "test"):
            measured = stage_results[field][split]
            rows = parts[task][split]
            samples = measured["samples"]
            assert len(samples) == len(rows) == measured["records"] == measured["examples"]
            labels = text_examples(rows, mode="sft")
            effective_tokens = sum(int((y != IGNORE).sum()) for _, y in labels)
            assert effective_tokens == measured["effective_tokens"], "loss 分母須由有效目標重算"
            assert math.isclose(measured["nll"], measured["nll_sum"] / effective_tokens, rel_tol=0, abs_tol=1e-12)
            matches, ended = 0, 0
            for sample, row in zip(samples, rows, strict=True):
                assert sample["messages"] == row["messages"][:-1]
                assert sample["expected"] == row["messages"][-1]["content"]
                ids = sample["generated_ids"]
                eos = tok.eos_id in ids
                raw = ids[: ids.index(tok.eos_id)] if eos else ids
                exact = raw == tok.encode(sample["expected"])
                assert sample["generated"] == tok.decode(raw)
                assert sample["exact"] == exact and sample["eos"] == eos
                assert len(ids) <= 32
                matches += exact
                ended += eos
            assert matches == measured["matches"]
            assert measured["exact_match"] == matches / len(rows)
            assert measured["eos_rate"] == ended / len(rows)
            evaluations[stage][field][split] = {
                "matches": matches,
                "records": len(rows),
                "effective_tokens": effective_tokens,
                "nll_sum": measured["nll_sum"],
                "nll_recomputed": measured["nll_sum"] / effective_tokens,
                "ended": ended,
            }

assert baseline["results"]["after"] == results["before"]["A_attributes"], "屬性底座的完整評估須相同"
training = after["training"]
assert after["source_records"] == {"A": 0, "B": 49}
assert training["records"] == 49 and training["steps"] == 500
assert training["final_loss"] < training["initial_loss"]
training_digest = digest(json.dumps(parts["arithmetic"]["train"], sort_keys=True, ensure_ascii=False).encode())
assert training_digest == training["records_sha256"]
assert evaluations["before"]["A_attributes"]["test"]["matches"] == 5
assert evaluations["b-only"]["A_attributes"]["test"]["matches"] == 0
assert evaluations["before"]["B_arithmetic"]["test"]["matches"] == 0
assert evaluations["b-only"]["B_arithmetic"]["test"]["matches"] == 0
before_nll = evaluations["before"]["B_arithmetic"]["test"]["nll_recomputed"]
after_nll = evaluations["b-only"]["B_arithmetic"]["test"]["nll_recomputed"]
assert f"{before_nll:.5f}" == "7.05820" and f"{after_nll:.5f}" == "4.57756"
example = next(
    sample for sample in after["A_attributes"]["test"]["samples"]
    if sample["expected"] == "circle" and sample["generated"] == "9" and sample["eos"]
)
assert f"{record['elapsed_seconds']:.1f}" == "23.5"

code_paths = [
    "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
    "scripts/course_experiments/run.py", "scripts/prepare_data.py",
    "tiny_perceptron/data.py", "tiny_perceptron/model.py",
    "course/training.md", "docs/course-experiments/plan.json",
    record_path, baseline_path,
]
code_versions = {}
for path in code_paths:
    current_hash = digest((ROOT / path).read_bytes())
    logged_hash = record["code_sha256"].get(path)
    code_versions[path] = {"sha256": current_hash, "recorded_sha256": logged_hash}
    if logged_hash:
        code_versions[path]["current_matches_record"] = logged_hash == current_hash
        if logged_hash != current_hash:
            attempted = subprocess.run(["git", "show", f"{record['revision']}:{path}"], cwd=ROOT, capture_output=True)
            code_versions[path]["record_revision_lookup"] = {
                "command": f"git show {record['revision']}:{path}",
                "exit_code": attempted.returncode,
                "stderr": attempted.stderr.decode(),
            }
            commits = subprocess.check_output(["git", "log", "--format=%H", "--all", "--", path], cwd=ROOT, text=True).splitlines()
            found = None
            for commit in commits:
                candidate = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True)
                if candidate.returncode == 0 and digest(candidate.stdout) == logged_hash:
                    found = commit
                    code_versions[path]["historical_commit"] = commit
                    code_versions[path]["historical_sha256"] = digest(candidate.stdout)
                    break
            assert found is not None, "以內容雜湊綁定的程式須可由 Git 取回"

cli_help = subprocess.check_output([sys.executable, "-m", "scripts.course_experiments.run", "--help"], cwd=ROOT, text=True)
cli_assets = subprocess.check_output([sys.executable, "-m", "scripts.course_experiments.run", "--list-assets", "sft_ablation"], cwd=ROOT, text=True)
assert "--experiment" in cli_help and "--device {cpu,cuda}" in cli_help
assert not cli_assets.strip(), "比較實驗本身沒有額外資料包"

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-15/audit.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "checks_passed": True,
    "section_stdout": stdout.getvalue(),
    "exercise": {"a_questions": namespace["a_questions"], "b_questions": namespace["b_questions"], "measurements": namespace["measurements"]},
    "data_regeneration": data,
    "arithmetic_population": {"ordered_questions": 64, "unordered_families": 36, "operand_min": 0, "operand_max": 7, "swapped_operands_same_family": True},
    "evaluations": evaluations,
    "baseline_evaluation_identical": True,
    "baseline_checkpoint": next(item for item in baseline["artifacts"] if item["path"] == "model.pt"),
    "training": {key: training[key] for key in ("initial_loss", "final_loss", "steps", "effective_tokens", "records", "records_sha256")},
    "digit_with_eos_example": example,
    "record_environment": {key: record[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status")},
    "code_versions": code_versions,
    "cli_help": cli_help,
    "cli_list_assets_stdout": cli_assets,
    "limitations": "僅重算保存紀錄、重建資料與執行清單程式；未載入權重、重訓、重跑 GPU 計時或獨立重現模型生成與 loss。",
}, ensure_ascii=False, indent=2))
