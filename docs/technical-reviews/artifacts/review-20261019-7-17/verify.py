"""離線核對成本、實驗紀錄、資料切分與命令入口，不啟動訓練。"""

import ast
import contextlib
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.course_experiments.run import experiment_spec
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


source = "docs/course-review-20261019/freeze-01/sources/7.17.md"
body = (ROOT / source).read_text()
code = re.search(r"```python\n(.*?)```", body, re.S)[1]
costs = {}
for unit_cost in (20, 5):
    tree = ast.parse(code)
    assignment = tree.body[1]
    require(isinstance(assignment, ast.Assign), "找不到每筆成本的指定")
    require(assignment.targets[0].elts[1].id == "checked_chat_cost", "成本變數不符")
    assignment.value.elts[1] = ast.Constant(value=unit_cost)
    ast.fix_missing_locations(tree)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(tree, source, "exec"), {})
    costs[str(unit_cost)] = out.getvalue()
require(costs["20"].splitlines() == ["文章整理成本 1000", "問答整理成本 1000", "若1000筆都逐筆核對問答 20000"], "原始成本結果不符")
require(costs["5"].splitlines() == ["文章整理成本 1000", "問答整理成本 250", "若1000筆都逐筆核對問答 5000"], "練習成本結果不符")
require(not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)), "成本算例出現額外匯入")

record_path = "docs/course-experiments/results/sft.json"
record = json.loads((ROOT / record_path).read_text())
results = record["results"]
branch = results["pretrain_then_sft"]
tokenizer = ByteTokenizer()
audits = {}
for name, report, expected in (
    ("before_sft", branch["before_sft"]["test"], 1),
    ("after_sft", branch["after_sft"]["test"], 7),
    ("direct_sft", results["after"]["test"], 5),
):
    rows = []
    for item in report["samples"]:
        ids = item["generated_ids"]
        raw = ids[:ids.index(tokenizer.eos_id)] if tokenizer.eos_id in ids else ids
        exact = raw == tokenizer.encode(item["expected"])
        require(exact == item["exact"], "逐題匹配旗標與原始 token 不符")
        require(tokenizer.decode(raw) == item["generated"], "解碼文字與紀錄不符")
        rows.append({"question": item["messages"][0]["content"], "expected": item["expected"], "generated": item["generated"], "exact": exact})
    matches = sum(row["exact"] for row in rows)
    require(len(rows) == report["records"] == 10, "測試題數不符")
    require(matches == report["matches"] == expected, "答對題數不符")
    require(report["exact_match"] == matches / len(rows), "匹配率的分母不符")
    audits[name] = {"matches_recomputed": matches, "records": len(rows), "rows": rows}
require([row["question"] for row in audits["before_sft"]["rows"]] == [row["question"] for row in audits["after_sft"]["rows"]] == [row["question"] for row in audits["direct_sft"]["rows"]], "比較支線的測試題不相同")

parts = split_records(generate_records("attributes-sft"), seed=record["seed"])
data = {}
for name, rows in parts.items():
    data[name] = {"records": len(rows), "families": sorted({row["family"] for row in rows}), "jsonl_sha256": hashlib.sha256("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()).hexdigest()}
    require(data[name]["jsonl_sha256"] == results["data"][name]["sha256"], "重建資料的 SHA 不符")
require(not (set(data["train"]["families"]) & set(data["test"]["families"])), "訓練和測試家族重疊")
train = parts["train"]
texts = [{"text": row["messages"][0]["content"] + row["messages"][1]["content"], "family": row["family"]} for row in train]
require(records_sha256(train) == results["training"]["records_sha256"] == branch["sft"]["records_sha256"], "SFT 訓練材料的 SHA 不符")
require(records_sha256(texts) == branch["pretraining"]["records_sha256"], "文字階段的材料 SHA 不符")
examples = text_examples(parts["test"], mode="sft")
effective = sum(int((y != -100).sum()) for _, y in examples)
require(effective == branch["after_sft"]["test"]["effective_tokens"] == 69, "測試有效目標分母不符")
require(branch["pretraining"]["steps"] == 250, "文字更新步數不符")
require(results["training"]["steps"] == branch["sft"]["steps"] == 900, "SFT 更新步數不符")
require(f'{record["elapsed_seconds"]:.1f}' == "31.7", "計時四捨五入不符")

env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
cli = []
for args in (
    ["scripts/fetch_training_assets.py", "--help"],
    ["-m", "scripts.course_experiments.run", "--help"],
    ["-m", "scripts.course_experiments.run", "--list-assets", "sft"],
):
    command = [str(ROOT / ".venv/bin/python"), "-B", *args]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, "命令入口核對失敗")
    cli.append({"command_argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
spec = experiment_spec("sft")
require(spec["module"] == "text" and spec["function"] == "run_sft" and spec["assets"] == ["ultrachat-sft"], "實驗派送或材料清單不符")

figure = "course/figures/rewrite-07-17-prepost-routes.svg"
receipt_path = "docs/course-review-20261019/figure-renders-04.json"
receipt = next(x for x in json.loads((ROOT / receipt_path).read_text())["records"] if x["page_id"] == "7.17")
require(sha(figure) == receipt["frozen_svg"]["sha256"], "SVG 的 SHA 不符")
require(sha(receipt["png"]["path"]) == receipt["png"]["sha256"], "PNG 的 SHA 不符")
svg = ET.parse(ROOT / figure).getroot()
require(svg.attrib["viewBox"] == "0 0 640 790", "SVG 畫布尺寸不符")
paths = [node.attrib["d"] for node in svg.findall("{http://www.w3.org/2000/svg}path")]
require(paths == ["M320 208V253", "M194 385V439", "M612 324H626V696H614"], "圖中流程箭頭路徑不符")

inspected = [source, "docs/course-review-20261019/freeze-04.json", figure, receipt_path, record_path, "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "scripts/course_experiments/run.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py", "docs/course-experiments/plan.json", "assets/training/manifest.json", "course/training.md", "course/chapters/07.md", "course/chapters/19.md", "docs/review-tools/section_facts.py"]
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B docs/technical-reviews/artifacts/review-20261019-7-17/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "result": "所有斷言通過；未啟動訓練、下載資料包或載入權重。",
    "cost_outputs": costs,
    "record_path": record_path,
    "recorded_environment": {k: record[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status")},
    "sample_audit": audits,
    "data_reconstruction": data,
    "training": {"pretrain_steps": branch["pretraining"]["steps"], "direct_sft_steps": results["training"]["steps"], "continued_sft_steps": branch["sft"]["steps"], "test_effective_tokens": effective},
    "code_hash_comparison": {p: {"recorded": h, "current": sha(p), "equal": sha(p) == h} for p, h in record["code_sha256"].items() if p in ["scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "scripts/course_experiments/run.py"]},
    "cli": cli,
    "experiment_spec": spec,
    "figure": {"receipt": receipt, "arrow_paths": paths, "viewed_with": "view_image"},
    "inspected_file_sha256": {p: sha(p) for p in inspected},
    "limits": "僅重算既有逐題紀錄與資料，不宣稱重現 GPU 訓練品質、硬體用量或速度；命令以 help、資產列舉及原始碼核對。"
}, ensure_ascii=False, indent=2))
