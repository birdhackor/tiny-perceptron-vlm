"""從凍結算例及已保存 logits 獨立重算，僅將核對結果寫到標準輸出。"""

import ast
import hashlib
import inspect
import json
import math
import platform
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.alignment import reliability_bins
from scripts.course_experiments.run import experiment_spec


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def close(actual, expected, tolerance=2e-7):
    assert abs(actual - expected) <= tolerance, f"數值不符：實際 {actual}，預期 {expected}"


def metrics(logits, labels, temperature):
    probabilities = (logits / temperature).softmax(-1)
    confidence = probabilities.max(-1).values
    correct = logits.argmax(-1).eq(labels)
    rows = reliability_bins(confidence, correct, bins=5)
    n = len(labels)
    return {
        "count": n,
        "correct": int(correct.sum()),
        "accuracy": int(correct.sum()) / n,
        "mean_confidence": float(confidence.mean()),
        "nll": float(F.cross_entropy(logits / temperature, labels)),
        "brier": float((probabilities - F.one_hot(labels, logits.shape[-1])).square().sum(-1).mean()),
        "ece": sum(row["count"] / n * abs(row["accuracy"] - row["confidence"]) for row in rows),
        "bins": rows,
        "probabilities": probabilities,
    }


def portable_metrics(value):
    return {key: item for key, item in value.items() if key != "probabilities"}


torch.set_num_threads(1)
raw = (ROOT / "docs/course-review-20261019/freeze-01/sources/9.10.md").read_text()
fences = re.findall(r"```python\n(.*?)```", raw, flags=re.S)
assert len(fences) == 2, "原文 Python 區塊數不符"
namespace = {"torch": torch}
exec(compile(fences[1], "凍結函式", "exec"), namespace)
frozen_function = namespace["reliability_bins"]
assert ast.dump(ast.parse(inspect.getsource(reliability_bins))) == ast.dump(ast.parse(fences[1])), "函式節錄與實作不符"

four = torch.tensor([0.55, 0.65, 0.85, 0.95])
four_correct = torch.tensor([1, 0, 1, 0])
five = torch.cat((four, torch.tensor([0.25])))
five_correct = torch.cat((four_correct, torch.tensor([1])))
toy = {}
for name, confidence, correct, expected in (
    ("four", four, four_correct, 0.25),
    ("exercise", five, five_correct, 0.35),
):
    rows = reliability_bins(confidence, correct, bins=2)
    assert rows == frozen_function(confidence, correct, bins=2), "凍結函式與匯入函式的結果不符"
    value = sum(row["count"] / len(confidence) * abs(row["accuracy"] - row["confidence"]) for row in rows)
    close(value, expected)
    toy[name] = {"bins": rows, "ece": value}

boundary = torch.tensor([0.0, 0.5, 1.0])
boundary_rows = reliability_bins(boundary, torch.tensor([1, 0, 1]), bins=2)
assert [row["count"] for row in boundary_rows] == [1, 2], "端點歸箱不符"
assert inspect.signature(reliability_bins).parameters["bins"].default == 5, "預設箱數不符"
mask_details = []
for i in range(2):
    left, right = i / 2, (i + 1) / 2
    selected = (four >= left) & (four <= right if i == 1 else four < right)
    mask_details.append({"mask": selected.tolist(), "any": bool(selected.any()), "count": int(selected.sum())})
toy["boundary"] = boundary_rows
toy["mask_details"] = mask_details
toy["default_bins"] = len(reliability_bins(torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]), torch.ones(5)))
toy["nll_examples"] = {"p_0.8": -math.log(0.8), "p_0.2": -math.log(0.2)}
toy["brier_example"] = 0.8**2 + (0.2 - 1)**2
toy["ece_by_bin_count"] = {
    str(n): sum(row["count"] / len(four) * abs(row["accuracy"] - row["confidence"]) for row in reliability_bins(four, four_correct, bins=n))
    for n in (1, 2, 4, 10)
}
close(toy["brier_example"], 1.28)

experiment_path = "docs/course-experiments/results/encoders.json"
record = json.loads((ROOT / experiment_path).read_bytes())
spec = experiment_spec("encoders")
assert spec["module"] == "modalities" and spec["function"] == "run_encoders", "實驗名稱的程式入口不符"
rerun_command = ".venv/bin/python -m scripts.course_experiments.run --experiment encoders --device cuda"
assert rerun_command in (ROOT / "course/training.md").read_text(), "訓練頁未包含重跑命令"
results = {}
for modality in ("vision", "audio"):
    saved = record["results"][modality]
    calibration = saved["calibration"]
    manifest = saved["data"]
    classes = saved["classes"]
    assert calibration["selection_split"] == "validation only", "選溫資料用途不符"
    assert calibration["temperature_grid"] == [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0], "預定網格不符"
    split_families = {}
    for split, data in manifest["splits"].items():
        rows = data["records"]
        assert data["count"] == len(rows), "資料筆數不符"
        raw_records = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
        assert hashlib.sha256(raw_records).hexdigest() == data["sha256"], "資料 manifest 雜湊不符"
        split_families[split] = {row["family"] for row in rows}
        if modality == "audio":
            assert all(row["answer"] == ("high" if row["frequency"] > 300 else "low") for row in rows), "音高標記不符"
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not split_families[first] & split_families[second], "題目家族在資料切分間重疊"

    validation = calibration["validation"]
    validation_logits = torch.tensor(validation["logits"], dtype=torch.float32)
    validation_labels = torch.tensor(validation["labels"], dtype=torch.int64)
    selection = []
    for candidate in calibration["selection"]:
        nll = float(F.cross_entropy(validation_logits / candidate["temperature"], validation_labels))
        close(nll, candidate["validation_nll"])
        selection.append({"temperature": candidate["temperature"], "validation_nll": nll})
    chosen = min(selection, key=lambda row: row["validation_nll"])["temperature"]
    assert chosen == calibration["chosen_temperature"] == 0.5, "最低驗證代價選出的溫度不符"
    observed = {}
    for split in ("validation", "test"):
        data = calibration[split]
        logits = torch.tensor(data["logits"], dtype=torch.float32)
        labels = torch.tensor(data["labels"], dtype=torch.int64)
        rows = manifest["splits"][split]["records"]
        assert list(logits.shape) == [len(labels), len(classes)], "logits 維度不符"
        assert len(rows) == data["count"] == len(labels), "評估分母不符"
        assert labels.tolist() == [classes.index(row["answer"]) for row in rows], "評估標記不符"
        assert data["families"] == [row["family"] for row in rows], "評估題目家族不符"
        assert torch.equal(logits.argmax(-1), (logits / chosen).argmax(-1)), "正溫度改變這批分類答案"
        observations = {}
        all_metrics = {}
        for condition, temperature in (("original", 1.0), ("calibrated", chosen)):
            calculated = metrics(logits, labels, temperature)
            all_metrics[condition] = calculated
            for key in ("count", "correct", "accuracy", "mean_confidence", "nll", "brier", "ece"):
                tolerance = max(1e-14, abs(data[condition][key]) * 1e-4) if key == "brier" else 2e-7
                close(calculated[key], data[condition][key], tolerance)
            torch.testing.assert_close(calculated["probabilities"], torch.tensor(data[condition]["probabilities"]), atol=2e-7, rtol=0)
            assert logits.argmax(-1).tolist() == data[condition]["predicted_labels"], "保存的預測標記不符"
            nonempty = [row for row in data[condition]["bins"] if row["count"]]
            assert len(nonempty) == len(calculated["bins"]), "非空箱數不符"
            for actual, expected in zip(calculated["bins"], nonempty, strict=True):
                assert actual["count"] == expected["count"], "箱內分母不符"
                close(actual["confidence"], expected["mean_confidence"])
                close(actual["accuracy"], expected["accuracy"])
            observations[condition] = portable_metrics(calculated)
        correct = logits.argmax(-1).eq(labels)
        before = all_metrics["original"]["probabilities"][torch.arange(len(labels)), labels]
        after = all_metrics["calibrated"]["probabilities"][torch.arange(len(labels)), labels]
        change = -(after.log() - before.log())
        observations["nll_change_by_correctness"] = {
            "correct_sum": float(change[correct].sum()),
            "incorrect_sum": float(change[~correct].sum()),
            "total_sum": float(change.sum()),
        }
        observations["errors"] = [
            {"index": i, "family": data["families"][i], "true_probability_before": float(before[i]), "true_probability_after": float(after[i])}
            for i in range(len(labels)) if not bool(correct[i])
        ]
        observed[split] = observations
    results[modality] = {
        "classes": classes,
        "config": saved["config"],
        "split_counts": {key: value["count"] for key, value in manifest["splits"].items()},
        "families": {key: sorted(value) for key, value in split_families.items()},
        "training_steps_recorded": saved["training"]["steps"],
        "selection": selection,
        "chosen_temperature": chosen,
        **observed,
    }

displayed = {
    "vision": {"original": {"ece": (5, 0.02858), "brier": (6, 0.001485)}, "calibrated": {"ece": (6, 0.000572), "brier": (9, 0.000000859)}},
    "audio": {"original": {"ece": (5, 0.13186), "brier": (5, 0.20207), "mean_confidence": (4, 0.8796), "nll": (4, 0.2776)}, "calibrated": {"ece": (5, 0.14154), "brier": (5, 0.25337), "mean_confidence": (4, 0.9273), "nll": (4, 0.3465)}},
}
for modality, conditions in displayed.items():
    for condition, fields in conditions.items():
        for field, (digits, expected) in fields.items():
            assert round(results[modality]["test"][condition][field], digits) == expected, "正文顯示精度與重算不符"

audio = results["audio"]["test"]["original"]
gap = audio["mean_confidence"] - audio["accuracy"]
assert audio["ece"] > abs(gap), "這批資料的分箱誤差應大於整體平均差"

ns = {"s": "http://www.w3.org/2000/svg"}
figures = {}
audio_path = "course/figures/audio_calibration_test.svg"
svg = ET.parse(ROOT / audio_path).getroot()
bars = [item for item in svg.findall("s:rect", ns) if item.attrib.get("fill") == "#357ba9"]
lines = svg.findall("s:line", ns)
for bar, confidence in zip(bars, (0.8796, 0.9273), strict=True):
    close(float(bar.attrib["width"]) / 560, confidence, 1e-12)
for line in lines:
    close((float(line.attrib["x1"]) - 32) / 560, 11 / 14, 1e-12)
figures[audio_path] = {"bar_ratios": [float(item.attrib["width"]) / 560 for item in bars], "accuracy_ratios": [(float(item.attrib["x1"]) - 32) / 560 for item in lines]}
bin_path = "course/figures/rewrite-09-10-calibration-bins.svg"
point = ET.parse(ROOT / bin_path).getroot().find("s:circle", ns)
xy = [(float(point.attrib["cx"]) - 130) / 350, (567 - float(point.attrib["cy"])) / 350]
assert xy == [0.75, 0.5], "圖中點座標不符"
figures[bin_path] = {"point": xy}
receipts = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())
for collection in receipts.values():
    if isinstance(collection, list):
        for item in collection:
            if isinstance(item, dict) and item.get("figure") in figures and item.get("page_id") == "9.10":
                assert sha(item["figure"]) == item["frozen_svg"]["sha256"], "SVG 凍結雜湊不符"
                assert sha(item["png"]["path"]) == item["png"]["sha256"], "PNG 收據雜湊不符"
                figures[item["figure"]]["render_receipt"] = item["png"]
for item in figures.values():
    assert "render_receipt" in item, "圖像算繪收據缺漏"

inputs = ["docs/course-review-20261019/freeze-01/sources/9.10.md", "docs/course-review-20261019/freeze-04.json", experiment_path, "tiny_perceptron/alignment.py", "scripts/course_experiments/modalities.py", "scripts/course_experiments/run.py", "docs/course-experiments/plan.json", "course/training.md", *figures]
for path in ("tiny_perceptron/alignment.py", "scripts/course_experiments/modalities.py"):
    assert sha(path) == record["code_sha256"][path], "現行程式與實驗紀錄版本不符"
output = {
    "result": "全部核對相符",
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-10/verify_calibration.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "dtype": "float32", "platform": platform.platform()},
    "input_sha256": {path: sha(path) for path in inputs},
    "original_run_environment": {key: record[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "evidence_status")},
    "tolerance": "浮點重算與已存結果通常絕對差至多 2e-7；Brier 使用 max(1e-14, |已存值|×1e-4)，另逐項確認所有正文顯示位數。端點計數、argmax、資料標記與 SHA 精確相等；圖中比例容許 1e-12。",
    "function_matches_frozen_excerpt": True,
    "experiment_entry": {"spec": spec, "documented_command": rerun_command, "training_executed": False},
    "toy": toy,
    "experiments": results,
    "audio_original_global_confidence_minus_accuracy": gap,
    "displayed_metrics_match": True,
    "figures": figures,
    "scope": "重算已保存 logits，核對程式與資料切分；未重新訓練、未讀取私有模型權重，不能外推到自然資料或生成回答。",
}
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
