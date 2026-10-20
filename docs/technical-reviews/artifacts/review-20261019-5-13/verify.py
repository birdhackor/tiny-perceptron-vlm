"""核對凍結教材的算例、實驗紀錄及原始來源；不執行訓練。"""

import ast
import contextlib
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from tiny_perceptron.data import IGNORE, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha((ROOT / path).read_bytes())


def plain_html(raw):
    text = raw.decode("utf-8")
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", text, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", text)))


torch.set_num_threads(1)
frozen_path = "docs/course-review-20261019/freeze-01/sources/5.13.md"
frozen = (ROOT / frozen_path).read_bytes()
expected_sha = "b605ca3c39976c1d3b5da3efbf1ee74d7e0416c17ac6071ff65a36c59fdf6aa0"
assert sha(frozen) == expected_sha
page = next(
    item
    for item in json.loads((ROOT / "docs/course-review-20261019/freeze-04.json").read_bytes())["pages"]
    if item["page_id"] == "5.13"
)
assert page["source_sha256"] == expected_sha and page["figures_sha256"] == {}
text = frozen.decode("utf-8")
fence = re.search(r"```python\n(.*?)```", text, re.S).group(1)
result_path = "docs/course-experiments/results/text_foundation.json"
record = json.loads((ROOT / result_path).read_bytes())
scaling = record["results"]["scaling"]
receipt = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-13/verify.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "platform": platform.platform(),
    },
    "source_sha256": sha(frozen),
    "figure_sha256": {},
    "input_hashes": {
        name: file_sha(name)
        for name in (
            frozen_path,
            "docs/course-review-20261019/freeze-04.json",
            result_path,
            "scripts/course_experiments/common.py",
            "scripts/course_experiments/text.py",
            "scripts/course_experiments/run.py",
            "tiny_perceptron/model.py",
            "tiny_perceptron/data.py",
            "tiny_perceptron/attention.py",
            "tiny_perceptron/modern.py",
            "course/training.md",
        )
    },
}

parameters = {}
for width, expected in ((8, 6104), (16, 13744), (32, 33632)):
    model = TinyLM(ModelConfig(width=width))
    counted = sum(parameter.numel() for parameter in model.parameters())
    assert counted == expected == model.description()["parameters"] == 12 * width**2 + 667 * width
    parameters[str(width)] = {
        "actual": counted,
        "formula": "12*w*w + 667*w",
        "configuration": model.description()["config"],
        "named_parameters": {name: parameter.numel() for name, parameter in model.named_parameters()},
    }
receipt["parameters"] = parameters

plans = []
for values, expected in (("4, 8, 100", 3200), ("4, 4, 200", 3200), ("4, 4, 100", 1600)):
    code = fence.replace("4, 8, 100", values)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "frozen-plan", "exec"), {})
    rows = [ast.literal_eval(line) for line in output.getvalue().splitlines()]
    assert len(rows) == 4
    assert all(row["有效目標預算"] == expected and row["品質"] == "尚待訓練與量測" for row in rows)
    plans.append({"assignment": values, "rows": rows})
receipt["plan_executions"] = plans
receipt["plan_calculation"] = {
    "baseline": 4 * 8 * 100,
    "required_updates": 3200 // (4 * 4),
    "shortened_budget": 4 * 4 * 100,
    "planned_average_record_exposures": {"16": 4 * 100 / 16, "64": 4 * 100 / 64},
}

linear = []
for width in (8, 16):
    layer = torch.nn.Linear(width, width, bias=False, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(1, width * width + 1, dtype=torch.float64).reshape(width, width))
    x = torch.arange(1, width + 1, dtype=torch.float64)
    manual = [sum(float(layer.weight[m, d]) * float(x[d]) for d in range(width)) for m in range(width)]
    assert layer(x).tolist() == manual
    linear.append(
        {
            "input_features": width,
            "output_features": width,
            "weight_shape": list(layer.weight.shape),
            "multiplications": width * width,
            "minimal_reduction_additions": width * (width - 1),
            "exact_dot_product_operations_without_bias": width * (2 * width - 1),
            "multiply_accumulate_estimate": 2 * width * width,
            "scope": "每個位置的密集線性前向估算；不是完整模型訓練成本或耗時。",
        }
    )
receipt["linear"] = linear

pool = [
    {
        "text": f"object={index};color={'red' if index % 2 else 'blue'};shape={'circle' if index % 3 else 'square'}.",
        "family": str(index),
    }
    for index in range(80)
]
assert pool[67]["text"] == "object=67;color=red;shape=circle."
parts = split_records(pool, seed=42)
assert [len(parts[key]) for key in ("train", "validation", "test")] == [64, 8, 8]
families = {key: {row["family"] for row in values} for key, values in parts.items()}
assert not (families["train"] & families["validation"] or families["train"] & families["test"] or families["validation"] & families["test"])
splits = {}
for split, rows in parts.items():
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    expected = record["results"]["scaling_data"][split]
    assert sha(serialized) == expected["sha256"]
    examples = text_examples(rows)
    targets = sum(int((labels != IGNORE).sum()) for _, labels in examples)
    assert targets == sum(len(row["text"].encode("utf-8")) + 1 for row in rows)
    splits[split] = {
        "records": len(rows),
        "families": [row["family"] for row in rows],
        "serialized_sha256": sha(serialized),
        "effective_targets_once": targets,
    }
receipt["splits"] = splits

schedules = {}
for count in (16, 64):
    rows = parts["train"][:count]
    examples = text_examples(rows)
    sampler = random.Random(42)
    schedule = [sampler.choices(range(count), k=4) for _ in range(150)]
    actual = 0
    for selected in schedule:
        _, labels, _ = pad_batch([examples[index] for index in selected])
        actual += int((labels != IGNORE).sum())
    independent = sum(len(rows[index]["text"].encode("utf-8")) + 1 for batch in schedule for index in batch)
    assert actual == independent == (20765 if count == 16 else 20623)
    frequencies = [sum(batch.count(index) for batch in schedule) for index in range(count)]
    schedules[str(count)] = {
        "updates": 150,
        "batch_size": 4,
        "record_draws": 600,
        "average_draws_per_record": 600 / count,
        "minimum_draws": min(frequencies),
        "maximum_draws": max(frequencies),
        "effective_targets": actual,
        "training_evaluation_targets_once": sum(len(labels) for _, labels in examples),
        "records_sha256": records_sha256(rows),
        "schedule_sha256": sha(json.dumps(schedule).encode()),
    }
    for width in (16, 32):
        training = scaling[f"scaling-w{width}-n{count}"]["training"]
        assert training["effective_tokens"] == actual
        assert training["records_sha256"] == records_sha256(rows)
        assert training["steps"] == 150 and training["records"] == count
receipt["sampling_replay"] = schedules

table_rows = []
for line in text.splitlines():
    if re.match(r"^\| (16|32) \| (16|64)篇", line):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        width, count = int(cells[0]), int(cells[1].removesuffix("篇"))
        archived = scaling[f"scaling-w{width}-n{count}"]
        values = [archived["training"]["final_loss"], archived["evaluation"]["validation"]["nll"], archived["evaluation"]["test"]["nll"]]
        assert int(cells[2].replace(",", "")) == archived["training"]["effective_tokens"]
        assert cells[3:] == [format(value, ".5f") for value in values]
        assert archived["parameters"] == parameters[str(width)]["actual"]
        evaluation = {}
        for split in ("validation", "test"):
            item = archived["evaluation"][split]
            assert item["effective_tokens"] == splits[split]["effective_targets_once"]
            computed = item["nll_sum"] / item["effective_tokens"]
            assert math.isclose(computed, item["nll"], rel_tol=0, abs_tol=1e-15)
            evaluation[split] = {"nll": item["nll"], "sum": item["nll_sum"], "denominator": item["effective_tokens"], "recomputed_mean": computed}
        table_rows.append({"width": width, "training_records": count, "archived_train_loss": values[0], "printed_loss_values": cells[3:], "evaluation": evaluation})
assert len(table_rows) == 4
receipt["archived_metrics"] = table_rows
receipt["comparisons"] = {
    str(width): {
        "validation_lower_with_more_records": scaling[f"scaling-w{width}-n64"]["evaluation"]["validation"]["nll"] < scaling[f"scaling-w{width}-n16"]["evaluation"]["validation"]["nll"],
        "training_higher_with_more_records": scaling[f"scaling-w{width}-n64"]["training"]["final_loss"] > scaling[f"scaling-w{width}-n16"]["training"]["final_loss"],
    }
    for width in (16, 32)
}
assert all(all(comparisons.values()) for comparisons in receipt["comparisons"].values())
probe_logits = torch.zeros(2, 3, 5)
probe_labels = torch.tensor([[0, 1, IGNORE], [2, IGNORE, IGNORE]])
total, effective = loss_sum(probe_logits, probe_labels)
assert effective.item() == 3 and abs(total.item() / 3 - math.log(5)) < 1e-6
receipt["loss_probe"] = {"valid_targets": effective.item(), "sum": total.item(), "mean": total.item() / effective.item(), "expected_mean": math.log(5), "dtype": str(probe_logits.dtype), "tolerance": 1e-6}
receipt["historical_run"] = {key: record[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status", "status")}
receipt["historical_code_hash_comparison"] = {
    name: {"recorded": record["code_sha256"][name], "current": file_sha(name)}
    for name in ("scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py")
}
historical = subprocess.run(["git", "show", f"{record['revision']}:scripts/course_experiments/common.py"], cwd=ROOT, capture_output=True, text=True)
receipt["historical_commit_lookup"] = {"argv": historical.args, "exit_code": historical.returncode, "stderr": historical.stderr.strip()}
receipt["checkpoint_availability"] = {
    name: (ROOT / "checkpoints/course/text_foundation" / f"{name}.pt").is_file() for name in scaling
}
receipt["measurement_scope"] = "本次核對原始結果紀錄、表格轉錄、資料切分與抽樣計數，沒有訓練或從 checkpoint 獨立重算模型 loss；當次 GPU 測量由 complete_run 紀錄支持。"

help_run = subprocess.run([sys.executable, "-m", "scripts.course_experiments.run", "--help"], cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, check=True)
training_doc = (ROOT / "course/training.md").read_text()
assert ".venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda" in training_doc
receipt["reproduction_entry"] = {"command": help_run.args, "stdout": help_run.stdout, "stderr": help_run.stderr, "exit_code": help_run.returncode, "training_command_present": True, "training_executed": False}

commit = str(torch.version.git_version)
external_specs = [
    {
        "id": "linear",
        "url": f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/nn/modules/linear.py",
        "version": f"PyTorch {torch.__version__}, commit {commit}",
        "locator": "Linear 的 docstring、weight 建構與 forward",
        "needles": ["y = xA^T + b", "(out_features, in_features)"],
        "details": "原始碼確認 Linear 權重的輸出軸在前、輸入軸在後；每個輸出由整個輸入向量加權求和，再加 bias。",
    },
    {
        "id": "cross_entropy",
        "url": f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/nn/modules/loss.py",
        "version": f"PyTorch {torch.__version__}, commit {commit}",
        "locator": "CrossEntropyLoss 的 class-index target 公式與 reduction",
        "needles": ["this loss also accepts this class index", "ignore_index", "sum"],
        "details": "已讀取 class-index target 的負 log-softmax 公式、ignore_index 的指示函數及 mean/sum 分支；本專案沒有 class weights，總和除以有效位置數符合該契約。",
    },
    {
        "id": "factorial",
        "url": "https://www.itl.nist.gov/div898/handbook/pri/section3/pri3331.htm",
        "version": "NIST/SEMATECH e-Handbook, Two-level full factorial designs；讀取當日版本",
        "locator": "Standard Order for a 2^k Level Factorial Design；Orthogonality Properties",
        "needles": ["We can readily generalize", "2-level full factorial with k factors"],
        "details": "已讀取完整設計矩陣與各因子正交性說明；將兩個因子各取兩個水準有四種組合，可在固定另一因子時比較變化。這不能排除單次隨機性的影響。",
    },
    {
        "id": "scaling_laws",
        "url": "https://arxiv.org/html/2001.08361v1",
        "version": "Kaplan et al., Scaling Laws for Neural Language Models, arXiv:2001.08361v1 (2020-01-23)",
        "locator": "Abstract；Notation；Parameter and Compute Scaling of Transformers 的運算表與估算公式；Caveats",
        "needles": ["We study empirical scaling laws", "where the factor of two comes from the multiply-accumulate operation used in matrix multiplication."],
        "details": "已讀取摘要、符號定義、參數與運算表、前向與訓練運算估算及 Caveats。論文把品質、模型大小、資料量和 compute 的關係當作有條件的經驗規律；矩陣乘法按乘加兩次運算估算，忽略 bias 等次要項。論文不替本節四個玩具設定提供外推保證，也不把參數量定義直接套用到本節含 embedding 的總參數數目。",
    },
]
external = []
for spec in external_specs:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", spec["url"]]
    fetched = subprocess.run(command, capture_output=True, check=True)
    readable = plain_html(fetched.stdout) if spec["id"] in {"factorial", "scaling_laws"} else fetched.stdout.decode("utf-8")
    for needle in spec["needles"]:
        assert needle in readable, (spec["id"], needle)
    external.append({**spec, "curl_argv": command, "exit_code": fetched.returncode, "accessed_on": "2026-10-20", "response_sha256": sha(fetched.stdout), "bytes_read": len(fetched.stdout), "checked_original": True})
receipt["external_originals"] = external
receipt["result"] = "所有已執行核對均符合凍結原文與結果紀錄；完整訓練及 GPU 效能未重測。"
destination = Path(__file__).with_name("verification.json")
with destination.open("x", encoding="utf-8") as handle:
    handle.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": receipt["result"], "parameters": {key: value["actual"] for key, value in parameters.items()}, "effective_targets": {key: value["effective_targets"] for key, value in schedules.items()}, "original_sources_read": len(external)}, ensure_ascii=False))
