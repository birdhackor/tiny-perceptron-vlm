"""Offline replay for lesson 9.9. No training, checkpoint execution, or downloads."""

import contextlib
import datetime
import hashlib
import io
import json
import math
import pathlib
import platform
import re
import subprocess
import sys

import torch
from torch.nn import functional as F

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.modalities import _audio_records, _media


def sha(data):
    return hashlib.sha256(data).hexdigest()


def section(path, lesson):
    text = (ROOT / path).read_text()
    match = re.search(r"^## " + re.escape(lesson) + r"\b.*?(?=^## |\Z)", text, re.M | re.S)
    assert match is not None
    return match.group()


freeze_path = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "9.9")
raw = (ROOT / page["snapshot"]).read_bytes()
body = raw.decode()
assert sha(raw) == page["source_sha256"]
assert body == section(page["source"], "9.9")
assert page["figures_sha256"] == {}
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 1
code = blocks[0]

embedded = []
for correct_index in [1, 0]:
    runnable = code if correct_index == 1 else code.replace("correct_index = 1", "correct_index = 0")
    namespace = {}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(runnable, f"frozen-9.9-correct-index-{correct_index}", "exec"), namespace)
    expected_bool = "False" if correct_index == 1 else "True"
    expected = (
        f"倍率 1.0 信心 0.6652 選擇 0 正確 {expected_bool}\n"
        f"倍率 10.0 信心 1.0 選擇 0 正確 {expected_bool}\n"
    )
    assert capture.getvalue() == expected
    embedded.append({"correct_index": correct_index, "stdout": capture.getvalue()})

scores = torch.tensor([[2.0, 1.0, 0.0]])
scale_results = []
for scale in [1.0, 10.0]:
    probability = (scores * scale).softmax(-1)
    confidence, prediction = probability.max(-1)
    stable_denominator = sum(math.exp(scale * (x - 2)) for x in [2, 1, 0])
    reference = 1 / stable_denominator
    assert abs(confidence.item() - reference) < 1e-7
    assert confidence.item() < 1
    assert probability.shape == (1, 3) and confidence.shape == prediction.shape == (1,)
    assert prediction.item() == 0
    assert torch.equal(probability, (scores / (1 / scale)).softmax(-1))
    scale_results.append({
        "scale": scale, "temperature": 1 / scale, "probabilities": probability.tolist()[0],
        "raw_confidence": confidence.item(), "float64_math_reference": reference,
        "rounded_to_4_places": round(confidence.item(), 4), "prediction": prediction.item(),
    })

result_path = "docs/course-experiments/results/encoders.json"
recorded_bytes = (ROOT / result_path).read_bytes()
recorded = json.loads(recorded_bytes)
audio = recorded["results"]["audio"]
calibration = audio["calibration"]
generated = _audio_records()
for split, rows in generated.items():
    assert rows == audio["data"]["splits"][split]["records"]
for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert {r["family"] for r in generated[first]}.isdisjoint(r["family"] for r in generated[second])
row = generated["test"][11]
assert row["frequency"] == 320 and row["amplitude"] == 0.5 and row["seconds"] == 0.12
assert row["answer"] == "high" and audio["classes"] == ["low", "high"]
wave = _media(row, type("Context", (), {"device": "cpu"})())[1]
assert len(wave) == 1920 and wave.dtype == torch.float32
assert abs(wave.abs().max().item() - 0.5) < 0.001

validation_logits = torch.tensor(calibration["validation"]["logits"], dtype=torch.float32)
validation_labels = torch.tensor(calibration["validation"]["labels"])
selection = []
for candidate in calibration["selection"]:
    value = candidate["temperature"]
    nll = F.cross_entropy(validation_logits / value, validation_labels).item()
    assert abs(nll - candidate["validation_nll"]) < 1e-8
    selection.append({"temperature": value, "validation_nll": nll})
chosen = min(selection, key=lambda item: item["validation_nll"])["temperature"]
assert chosen == calibration["chosen_temperature"] == 0.5

test_logits = torch.tensor(calibration["test"]["logits"], dtype=torch.float32)
test_labels = torch.tensor(calibration["test"]["labels"])
replay = []
for temperature, key in [(1.0, "original"), (chosen, "calibrated")]:
    probabilities = (test_logits / temperature).softmax(-1)
    confidence = probabilities.max(-1).values
    prediction = (test_logits / temperature).argmax(-1)
    recorded_part = calibration["test"][key]
    error = (confidence - torch.tensor(recorded_part["confidence"])).abs().max().item()
    assert error < 1e-7
    assert prediction.tolist() == recorded_part["predicted_labels"]
    assert torch.equal(prediction, test_logits.argmax(-1))
    assert prediction[11].item() == 0 and test_labels[11].item() == 1
    replay.append({
        "temperature": temperature, "count": len(test_labels),
        "correct": (prediction == test_labels).sum().item(),
        "max_abs_confidence_error_vs_record": error,
        "sample_11_confidence": confidence[11].item(), "sample_11_prediction": prediction[11].item(),
        "sample_11_correct": bool(prediction[11] == test_labels[11]),
    })

code_paths = ["scripts/course_experiments/modalities.py", "tiny_perceptron/multimodal.py"]
code_pins = {}
for path in code_paths:
    current = (ROOT / path).read_bytes()
    original = subprocess.run(["git", "show", f"{recorded['revision']}:{path}"], capture_output=True, check=True).stdout
    assert original == current
    if path in recorded["code_sha256"]:
        assert sha(current) == recorded["code_sha256"][path]
    code_pins[path] = {"sha256": sha(current), "identical_to_recorded_revision": True}

context_path = pathlib.Path(__file__).with_name("context-slices.json")
context = json.loads(context_path.read_bytes())
for item in context["slices"]:
    current = section(item["source"], item["lesson_id"])
    if item["selector"] == "whole_section":
        actual = current
    else:
        actual = current[current.index(item["start_marker"]):]
        if item.get("end_marker"):
            actual = actual[:actual.index(item["end_marker"])]
    assert actual == item["text"]
    assert sha(actual.encode()) == item["sha256"]

cli = subprocess.run(
    [sys.executable, "-m", "scripts.course_experiments.run", "--help"], cwd=ROOT,
    capture_output=True, text=True, check=True,
)
assert "--experiment ID" in cli.stdout and "--device {cpu,cuda}" in cli.stdout

output = {
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-9-v1/review-evidence.py",
    "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "torch_git": torch.version.git_version,
                    "platform": platform.platform(), "device": "cpu", "dtype": "float32"},
    "frozen_source_sha256": sha(raw), "embedded_code_sha256": sha(code.encode()),
    "embedded_program": embedded, "scale_results": scale_results,
    "recorded_experiment": {
        "source": result_path, "sha256": sha(recorded_bytes), "revision": recorded["revision"],
        "device": recorded["device"], "gpu": recorded["gpu"], "seed": recorded["seed"],
        "torch": recorded["torch_version"], "python": recorded["python_version"],
        "evidence_status": recorded["evidence_status"], "steps": audio["training"]["steps"],
        "effective_targets": audio["training"]["effective_targets"],
        "history_entries": len(audio["training"]["history"]),
        "weights_changed": audio["training"]["weights_changed"],
        "nonzero_gradient_seen": audio["training"]["nonzero_gradient_seen"],
    },
    "denominators": {"train": 16, "validation": 8, "test": 14, "seed": 42, "recorded_training_steps": 250,
                     "recorded_training_targets": 2000, "new_training_steps": 0},
    "recorded_test_row_11": row, "waveform": {"shape": list(wave.shape), "dtype": str(wave.dtype),
                                                "sample_rate": 16000, "max_abs": wave.abs().max().item()},
    "validation_selection_replay": selection, "selected_temperature": chosen,
    "test_replay": replay, "sample_11_logits": test_logits[11].tolist(),
    "sample_11_label": test_labels[11].item(), "code_pins": code_pins,
    "context_slices_match_current": True, "context_slices_sha256": sha(context_path.read_bytes()),
    "cli_help_stdout": cli.stdout, "new_training": False,
    "scope": "CPU replay of exact lesson code and existing saved logits; no new model training or generalization measurement.",
    "result": "All assertions passed.",
}
target = pathlib.Path(__file__).with_name("execution.json")
assert not target.exists(), "Evidence is immutable; choose a new version for a rerun."
target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(output, ensure_ascii=False, indent=2))
