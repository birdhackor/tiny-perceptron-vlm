"""Bounded CPU arithmetic and original-JSON audit; no model inference or training."""
import ast
import contextlib
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def capture(code):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "current-9.9-fence", "exec"), {})
    return out.getvalue()


fence = (BASE / "execution/fence-1.py").read_bytes()
original_out = capture(fence)
changed_out = capture(fence.replace(b"correct_index = 1", b"correct_index = 0"))
assert original_out == "倍率 1.0 信心 0.6652 選擇 0 正確 False\n倍率 10.0 信心 1.0 選擇 0 正確 False\n"
assert changed_out == original_out.replace("False", "True")
numeric = []
for scale in (1.0, 10.0):
    p = (torch.tensor([[2.0, 1.0, 0.0]]) * scale).softmax(-1)
    v, i = p.max(-1)
    reference = 1 / (1 + math.exp(-scale) + math.exp(-2 * scale))
    assert p.shape == (1, 3) and v.shape == i.shape == (1,)
    assert isinstance(v.item(), float) and isinstance(i.item(), int)
    assert not p.requires_grad and 0 < v.item() < 1
    assert abs(v.item() - reference) < 1e-7
    assert abs(p.sum(-1).item() - 1) < 1e-7 and i.item() == 0
    numeric.append({"scale": scale, "float32_confidence": v.item(), "math_reference": reference,
                    "probabilities": p.tolist(), "prediction": i.item(), "rounded": round(v.item(), 4)})
for temperature in (0.5, 1.0, 2.0, 5.0):
    logits = torch.tensor([[2.0, 1.0, 0.0], [-6.0, -7.0, -8.0]])
    assert torch.equal(logits.argmax(-1), (logits / temperature).argmax(-1))
    assert torch.allclose((logits / temperature).softmax(-1).sum(-1), torch.ones(2))

raw_result = (ROOT / "docs/course-experiments/results/encoders.json").read_bytes()
report = json.loads(raw_result)
audio = report["results"]["audio"]
calibration = audio["calibration"]
source_path = ROOT / "scripts/course_experiments/modalities.py"
raw_source = source_path.read_bytes()
tree = ast.parse(raw_source)
runner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_encoders")
metric_node = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "confidence_metrics")
records_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_audio_records")
# Execute only these original functions, with the exact closure variables supplied.
isolated = ast.Module(body=[records_node, metric_node], type_ignores=[])
namespace = {"torch": torch, "F": F, "classes": audio["classes"]}
exec(compile(isolated, str(source_path) + ":original-functions-only", "exec"), namespace)
splits = namespace["_audio_records"]()
metric = namespace["confidence_metrics"]
assert audio["classes"] == ["low", "high"]
assert audio["config"] == {"width": 16, "bands": 16}
families = {}
for split, rows in splits.items():
    stored = audio["data"]["splits"][split]
    assert rows == stored["records"] and len(rows) == stored["count"]
    serialized = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
    assert sha(serialized) == stored["sha256"]
    families[split] = {r["family"] for r in rows}
for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
    assert not families[first] & families[second]

def compare(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            compare(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected, strict=True):
            compare(x, y)
    elif isinstance(actual, float):
        assert abs(actual - expected) <= 1e-7, (actual, expected)
    else:
        assert actual == expected, (actual, expected)

selection = []
for temperature in calibration["temperature_grid"]:
    validation = calibration["validation"]
    logits = torch.tensor(validation["logits"])
    labels = torch.tensor(validation["labels"])
    selection.append({"temperature": temperature, "validation_nll": float(F.cross_entropy(logits / temperature, labels))})
compare(selection, calibration["selection"])
chosen = min(selection, key=lambda c: c["validation_nll"])["temperature"]
assert chosen == calibration["chosen_temperature"] == 0.5
recomputed = {}
for split in ("validation", "test"):
    stored = calibration[split]
    logits = torch.tensor(stored["logits"])
    labels = torch.tensor(stored["labels"])
    assert len(logits) == len(labels) == stored["count"] == len(splits[split])
    assert labels.tolist() == [audio["classes"].index(r["answer"]) for r in splits[split]]
    assert stored["families"] == [r["family"] for r in splits[split]]
    assert torch.equal(logits.argmax(-1), (logits / chosen).argmax(-1))
    recomputed[split] = {}
    for name, temperature in (("original", 1.0), ("calibrated", chosen)):
        result = metric(logits, labels, temperature)
        compare(result, stored[name])
        recomputed[split][name] = result
    assert audio[split]["correct"] == int((logits.argmax(-1) == labels).sum())
    for i, sample in enumerate(audio[split]["samples"]):
        assert sample["row"] == i and sample["family"] == splits[split][i]["family"]
        assert sample["target"] == audio["classes"][labels[i]]
        assert sample["predicted"] == audio["classes"][logits[i].argmax()]
        assert sample["correct"] == (sample["target"] == sample["predicted"])

index = next(i for i, row in enumerate(splits["test"])
             if row["frequency"] == 320 and row["amplitude"] == 0.5 and row["seconds"] == 0.12)
row = splits["test"][index]
assert row["answer"] == "high" and index == 11
old = recomputed["test"]["original"]
new = recomputed["test"]["calibrated"]
assert old["predicted_labels"][index] == new["predicted_labels"][index] == 0
assert round(old["confidence"][index], 4) == 0.7490
assert round(new["confidence"][index], 4) == 0.8990
assert audio["training"]["steps"] == len(audio["training"]["history"]) == 250
assert sum(h["effective_targets"] for h in audio["training"]["history"]) == audio["training"]["effective_targets"] == 2000
assert audio["training"]["weights_changed"] is True and audio["training"]["nonzero_gradient_seen"] is True
code_receipts = []
for path in ("scripts/course_experiments/modalities.py", "tiny_perceptron/multimodal.py"):
    current = (ROOT / path).read_bytes()
    old_revision = subprocess.run(["git", "show", report["revision"] + ":" + path], capture_output=True, check=True).stdout
    assert sha(current) == sha(old_revision) == report["code_sha256"][path]
    code_receipts.append({"path": path, "sha256": sha(current), "revision": report["revision"]})

result = {
    "status": "passed",
    "scope": "Original fence and label-only exercise; arithmetic and original saved logits/JSON recomputation. No weights loaded, inference performed, or training rerun.",
    "environment": {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__),
                    "torch_git": str(torch.version.git_version), "cuda_build": str(torch.version.cuda),
                    "cuda_available": str(torch.cuda.is_available()), "device": "cpu", "threads": str(torch.get_num_threads())},
    "original_stdout": original_out, "exercise_stdout": changed_out, "hand_formula_checks": numeric,
    "original_json_sha256": sha(raw_result), "historical_run": {k: report[k] for k in ("revision", "device", "seed", "torch_version", "status", "modal")},
    "original_code_receipts": code_receipts,
    "split_counts": {s: len(rows) for s, rows in splits.items()}, "split_families": {s: sorted(v) for s,v in families.items()},
    "selection": selection, "chosen_temperature": chosen,
    "recomputed_metrics": recomputed,
    "selected_error": {"index": index, "row": row, "logits": calibration["test"]["logits"][index],
                       "prediction": "low", "target": "high", "original_confidence": old["confidence"][index],
                       "calibrated_confidence": new["confidence"][index]},
    "tolerance": "Absolute 1e-7 for saved float32 probabilities, NLL and metrics; exact for labels, counts, data records and hashes. Printed confidence uses Python round(value,4).",
    "figures": "The current 9.9 section has no figure references; no image rendered or visual verification claimed.",
}
(BASE / "execution/bounded-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": result["status"], "split_counts": result["split_counts"], "chosen_temperature": chosen,
                  "selected_error": result["selected_error"], "code_hashes_match_recorded_revision": True}, ensure_ascii=False))
