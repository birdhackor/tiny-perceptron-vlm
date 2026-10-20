import contextlib
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.modalities import _audio_records, _media
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition):
    if not condition:
        raise AssertionError("實際結果與核對條件不符")


base = Path(__file__).parent
section = json.loads((base / "section-execution.json").read_text())
frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/9.9.md"
require(digest(frozen) == section["extraction"]["source_sha256"])
scores = torch.tensor([[2.0, 1.0, 0.0]])
toy = []
for scale in [1.0, 10.0]:
    probabilities = (scores * scale).softmax(-1)
    confidence, prediction = probabilities.max(-1)
    exact = 1 / (1 + math.exp(-scale) + math.exp(-2 * scale))
    require(abs(confidence.item() - exact) < 1e-7)
    require(probabilities.shape == (1, 3))
    require(confidence.shape == prediction.shape == (1,))
    require(prediction.item() == 0)
    require(confidence.item() < 1)
    require(abs(probabilities.sum().item() - 1) < 2e-7)
    require(not scores.requires_grad)
    toy.append({
        "scale": scale,
        "probabilities": probabilities.tolist(),
        "confidence": confidence.item(),
        "closed_form_float64": exact,
        "rounded": round(confidence.item(), 4),
        "prediction": prediction.item(),
        "correct_index_1": prediction.item() == 1,
        "correct_index_0": prediction.item() == 0,
        "probability_shape": list(probabilities.shape),
        "reduced_shape": list(confidence.shape),
        "confidence_python_type": type(confidence.item()).__name__,
        "prediction_python_type": type(prediction.item()).__name__,
    })
require([r["rounded"] for r in toy] == [0.6652, 1.0])
exercise_stdout = io.StringIO()
with contextlib.redirect_stdout(exercise_stdout):
    exec(section["code"].replace("correct_index = 1", "correct_index = 0"), {})
require(exercise_stdout.getvalue().count("True") == 2)
temperature_examples = []
for temperature in [0.1, 1.0, 10.0]:
    a = (scores / temperature).softmax(-1)
    b = (scores * (1 / temperature)).softmax(-1)
    require(torch.equal(a, b))
    require(a.argmax(-1).item() == 0)
    temperature_examples.append({"temperature": temperature, "maximum": a.max().item()})

record_path = ROOT / "docs/course-experiments/results/encoders.json"
record = json.loads(record_path.read_text())
audio = record["results"]["audio"]
calibration = audio["calibration"]
splits = _audio_records()
for split, rows in splits.items():
    require(rows == audio["data"]["splits"][split]["records"])
    require(len(rows) == audio["data"]["splits"][split]["count"])
families = {name: sorted({r["family"] for r in rows}) for name, rows in splits.items()}
for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    require(not (set(families[first]) & set(families[second])))
training = audio["training"]
require(training["steps"] == len(training["history"]) == 250)
require(sum(r["effective_targets"] for r in training["history"]) == training["effective_targets"] == 2000)
require(training["weights_changed"] and training["nonzero_gradient_seen"])
require(audio["classes"] == ["low", "high"])
row = splits["test"][11]
require(row["frequency"] == 320 and row["amplitude"] == 0.5 and row["seconds"] == 0.12)
require(row["answer"] == "high")
wave = _media(row, SimpleNamespace(device="cpu"))[1]
require(wave.shape == (1920,))
require(torch.isfinite(wave).all().item())

validation_logits = torch.tensor(calibration["validation"]["logits"])
validation_labels = torch.tensor(calibration["validation"]["labels"])
grid = []
for candidate in calibration["temperature_grid"]:
    loss = F.cross_entropy(validation_logits / candidate, validation_labels).item()
    saved = next(r["validation_nll"] for r in calibration["selection"] if r["temperature"] == candidate)
    require(abs(loss - saved) < 1e-6)
    grid.append({"temperature": candidate, "nll_recomputed": loss, "nll_recorded": saved})
chosen = min(grid, key=lambda r: r["nll_recomputed"])["temperature"]
require(chosen == calibration["chosen_temperature"] == 0.5)
require(calibration["selection_split"] == "validation only")

recomputed = {}
for split in ["validation", "test"]:
    saved = calibration[split]
    logits = torch.tensor(saved["logits"])
    labels = torch.tensor(saved["labels"])
    require(len(labels) == len(splits[split]))
    require(labels.tolist() == [audio["classes"].index(r["answer"]) for r in splits[split]])
    recomputed[split] = {}
    for name, temperature in [("original", 1.0), ("calibrated", chosen)]:
        p = (logits / temperature).softmax(-1)
        confidence, prediction = p.max(-1)
        error = (confidence - torch.tensor(saved[name]["confidence"])).abs().max().item()
        require(error < 2e-7)
        require(prediction.tolist() == saved[name]["predicted_labels"])
        require(torch.equal(prediction, logits.argmax(-1)))
        require(torch.allclose(p, torch.tensor(saved[name]["probabilities"]), atol=2e-7, rtol=0))
        recomputed[split][name] = {
            "count": len(labels),
            "correct": int((prediction == labels).sum()),
            "confidence": confidence.tolist(),
            "maximum_confidence_difference": error,
            "argmax_invariant": True,
        }

test = calibration["test"]
require(test["original"]["predicted_labels"][11] == test["calibrated"]["predicted_labels"][11] == 0)
require(test["labels"][11] == 1)
require(round(test["original"]["confidence"][11], 4) == 0.7490)
require(round(test["calibrated"]["confidence"][11], 4) == 0.8990)
binary_identity = test["original"]["confidence"][11] ** 2
binary_identity /= binary_identity + (1 - test["original"]["confidence"][11]) ** 2
require(abs(binary_identity - test["calibrated"]["confidence"][11]) < 2e-7)

paths = [
    "scripts/course_experiments/modalities.py",
    "tiny_perceptron/multimodal.py",
    "scripts/course_experiments/run.py",
    "course/training.md",
    "docs/course-experiments/results/encoders.json",
]
current_hashes = {p: digest(ROOT / p) for p in paths}
code_record_matches = {
    p: current_hashes[p] == record["code_sha256"][p]
    for p in paths if p in record["code_sha256"]
}
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-9/verify.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "device": "cpu",
        "dtype": str(scores.dtype),
        "platform": platform.platform(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    },
    "result": "全部斷言通過；未重跑訓練，只從保存的 logits 重算機率、選溫與分類。",
    "source_sha256": digest(frozen),
    "toy": toy,
    "exercise_stdout": exercise_stdout.getvalue(),
    "temperature_examples": temperature_examples,
    "experiment_file_sha256": digest(record_path),
    "recorded_run": {k: record[k] for k in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status", "status"]},
    "training": {k: training[k] for k in ["steps", "effective_targets", "weights_changed", "nonzero_gradient_seen", "parameters"]},
    "split_counts": {name: len(rows) for name, rows in splits.items()},
    "frequency_families": families,
    "sample_index": 11,
    "sample": row,
    "waveform_samples": wave.numel(),
    "sample_logits": test["logits"][11],
    "sample_label": test["labels"][11],
    "sample_confidence_original": test["original"]["confidence"][11],
    "sample_confidence_calibrated": test["calibrated"]["confidence"][11],
    "sample_confidence_binary_identity": binary_identity,
    "temperature_grid": grid,
    "chosen_temperature": chosen,
    "recomputed": recomputed,
    "repository_file_hashes": current_hashes,
    "current_code_matches_recorded_run": code_record_matches,
    "limitations": "歷史 GPU 訓練的發生與權重更新依實驗原始紀錄核對，未獨立重演；CPU 重算只核對已保存 logits 的後處理。資料為單 seed、小量合成單音；不外推成生成答案正確率或安全可靠度。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
