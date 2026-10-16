"""獨立核對9.10算例及歷史logits；不重訓模型。"""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices
from tiny_perceptron.alignment import reliability_bins


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_metrics(logits, labels, temperature):
    # 使用Python雙精度純量及原始logits，獨立於原報告的torch實作。
    probabilities = []
    for row in logits:
        shifted = [value / temperature - max(row) / temperature for value in row]
        exp_values = [math.exp(value) for value in shifted]
        denominator = sum(exp_values)
        probabilities.append([value / denominator for value in exp_values])
    predictions = [max(range(len(row)), key=row.__getitem__) for row in logits]
    confidence = [max(row) for row in probabilities]
    correct = [int(prediction == label) for prediction, label in zip(predictions, labels)]
    count = len(labels)
    bins = []
    for index in range(5):
        members = [i for i, value in enumerate(confidence) if min(int(value * 5), 4) == index]
        bins.append({
            "count": len(members),
            "mean_confidence": sum(confidence[i] for i in members) / len(members) if members else None,
            "accuracy": sum(correct[i] for i in members) / len(members) if members else None,
        })
    return {
        "count": count,
        "correct": sum(correct),
        "accuracy": sum(correct) / count,
        "mean_confidence": sum(confidence) / count,
        "ece": sum(row["count"] / count * abs(row["accuracy"] - row["mean_confidence"]) for row in bins if row["count"]),
        "brier": sum(sum((value - int(class_id == label)) ** 2 for class_id, value in enumerate(row)) for row, label in zip(probabilities, labels)) / count,
        "nll": -sum(math.log(row[label]) for row, label in zip(probabilities, labels)) / count,
        "bins": bins,
        "predictions": predictions,
    }


body = lesson_slices((ROOT / "course/chapters/09.md").read_text())["9.10"]
code = re.search(r"```python\n(.*?)```", body, re.S)[1]
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, "course/chapters/09.md#9.10", "exec"), {})
examples = {}
for name, confidence, correct, count in [
    ("original", [0.55, 0.65, 0.85, 0.95], [1, 0, 1, 0], 2),
    ("exercise", [0.55, 0.65, 0.85, 0.95, 0.25], [1, 0, 1, 0, 1], 2),
    ("boundaries", [0.0, 0.5, 1.0], [0, 1, 1], 2),
    ("bin_sensitivity", [0.55, 0.65, 0.85, 0.95], [1, 0, 1, 0], 10),
]:
    rows = reliability_bins(torch.tensor(confidence), torch.tensor(correct), bins=count)
    ece = sum(row["count"] / len(confidence) * abs(row["accuracy"] - row["confidence"]) for row in rows)
    examples[name] = {"bins": rows, "ece": ece}
assert abs(examples["original"]["ece"] - 0.25) < 1e-6
assert abs(examples["exercise"]["ece"] - 0.35) < 1e-6
assert [row["count"] for row in examples["boundaries"]["bins"]] == [1, 2]
assert abs(examples["bin_sensitivity"]["ece"] - 0.55) < 1e-6

report_path = ROOT / "docs/course-experiments/results/encoders.json"
original = json.loads(report_path.read_text())
historical = {}
for modality in ["vision", "audio"]:
    result = original["results"][modality]
    calibration = result["calibration"]
    selected = []
    validation = calibration["validation"]
    for temperature in calibration["temperature_grid"]:
        value = independent_metrics(validation["logits"], validation["labels"], temperature)["nll"]
        reported = next(row["validation_nll"] for row in calibration["selection"] if row["temperature"] == temperature)
        assert abs(value - reported) < 2e-6
        selected.append({"temperature": temperature, "nll": value})
    assert min(selected, key=lambda row: row["nll"])["temperature"] == calibration["chosen_temperature"] == 0.5
    checked = {}
    for split in ["validation", "test"]:
        data = calibration[split]
        checked[split] = {}
        for name in ["original", "calibrated"]:
            reported = data[name]
            metrics = independent_metrics(data["logits"], data["labels"], reported["temperature"])
            errors = {}
            for key in ["accuracy", "mean_confidence", "ece", "brier", "nll"]:
                errors[key] = abs(metrics[key] - reported[key])
                assert errors[key] < 2e-6, (modality, split, name, key, errors[key])
            assert metrics["correct"] == reported["correct"]
            assert metrics["count"] == reported["count"]
            assert [row["count"] for row in metrics["bins"]] == [row["count"] for row in reported["bins"]]
            checked[split][name] = {"recomputed": metrics, "absolute_errors": errors}
        assert checked[split]["original"]["recomputed"]["predictions"] == checked[split]["calibrated"]["recomputed"]["predictions"]
    historical[modality] = {
        "data_counts": {split: rows["count"] for split, rows in result["data"]["splits"].items()},
        "steps": result["training"]["history"][-1]["step"],
        "classes": result["classes"],
        "selection_recomputed": selected,
        "chosen_temperature": 0.5,
        "metrics": checked,
    }
audio = original["results"]["audio"]
audio_test = audio["calibration"]["test"]
wrong_rows = [i for i, (prediction, label) in enumerate(zip(audio_test["original"]["predicted_labels"], audio_test["labels"])) if prediction != label]
audio_frequencies = {split: sorted({row["frequency"] for row in value["records"]}) for split, value in audio["data"]["splits"].items()}
historical["audio"]["frequencies"] = audio_frequencies
historical["audio"]["wrong_test_rows"] = [{"row": i, "family": audio_test["families"][i]} for i in wrong_rows]
assert len(wrong_rows) == 3
numeric = {"negative_log_0.8": -math.log(0.8), "negative_log_0.2": -math.log(0.2), "brier_example": 0.8**2 + (0.2 - 1)**2,
           "audio_global_gap": audio_test["original"]["mean_confidence"] - audio_test["original"]["accuracy"]}
assert abs(numeric["brier_example"] - 1.28) < 1e-12
hashes = {path: digest(ROOT / path) for path in ["tiny_perceptron/alignment.py", "scripts/course_experiments/modalities.py"]}
assert all(value == original["code_sha256"][path] for path, value in hashes.items())

output = {
    "reviewer_task": "/root/tech_20261016_9_10",
    "command": "uv run --no-sync python docs/technical-reviews/artifacts/revision-20261016/9.10-verify.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "result": "All assertions passed; exact lesson snippet executed; historical logits independently recomputed without training.",
    "lesson_stdout": stdout.getvalue(),
    "examples": examples,
    "numeric": numeric,
    "historical_report": {"path": report_path.relative_to(ROOT).as_posix(), "sha256": digest(report_path), "revision": original["revision"],
                          "seed": original["seed"], "device": original["device"], "gpu": original["gpu"], "torch": original["torch_version"], "python": original["python_version"], "code_hashes_match_current": hashes,
                          "scope": "Original training/evaluation was recorded on NVIDIA L4. This review only recomputes saved logits on CPU and does not rerun the training or assert GPU performance."},
    "historical_recomputation": historical,
}
destination = ROOT / "docs/technical-reviews/artifacts/revision-20261016/9.10-execution.json"
destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(output["result"])
print(stdout.getvalue())
print("Saved", destination.relative_to(ROOT))
