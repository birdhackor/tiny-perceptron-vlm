"""Replay only saved measurements through exact original helpers; no model inference/training."""
import ast
import hashlib
import json
import textwrap
from pathlib import Path
import torch
from torch.nn import functional as F

assert torch.version.cuda is None and not torch.cuda.is_available()
ARTIFACT = Path(__file__).resolve().parents[1]
record = json.loads((ARTIFACT / "inputs/encoders-selected-pointers.json").read_text())
v = record["selected_values"]
code = (ARTIFACT / "code/modalities.py").read_text()
tree = ast.parse(code)
names = ["_audio_records", "_hash", "confidence_metrics"]
namespace = {"torch": torch, "F": F, "classes": v["/results/audio/classes"], "json": json, "hashlib": hashlib}
extracted = []
for name in names:
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    segment = textwrap.dedent(ast.get_source_segment(code, node))
    extracted.append(segment)
    exec(compile(segment, f"original-modalities.py:{node.lineno}", "exec"), namespace)
(ARTIFACT / "code/exact-original-replayed-helpers.py").write_text("\n\n".join(extracted) + "\n")

splits = namespace["_audio_records"]()
for split, rows in splits.items():
    base = f"/results/audio/data/splits/{split}"
    assert rows == v[base + "/records"]
    assert len(rows) == v[base + "/count"]
    assert namespace["_hash"](rows) == v[base + "/sha256"]
    assert all(r["answer"] == ("high" if r["frequency"] > 300 else "low") for r in rows)
    print("manifest", split, "count", len(rows), "original_hash_exact", namespace["_hash"](rows))
for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert {r["family"] for r in splits[first]}.isdisjoint({r["family"] for r in splits[second]})

base = "/results/audio/calibration/validation"
validation_logits = torch.tensor(v[base + "/logits"], dtype=torch.float32)
validation_labels = torch.tensor(v[base + "/labels"], dtype=torch.long)
selection = []
for t in v["/results/audio/calibration/temperature_grid"]:
    nll = F.cross_entropy(validation_logits / t, validation_labels).item()
    recorded = next(row["validation_nll"] for row in v["/results/audio/calibration/selection"] if row["temperature"] == t)
    assert abs(nll - recorded) <= 1e-6
    selection.append({"temperature": t, "validation_nll_recomputed": nll})
temperature = min(selection, key=lambda row: row["validation_nll_recomputed"])["temperature"]
assert temperature == v["/results/audio/calibration/chosen_temperature"] == 0.5
print("validation_only_temperature_selection", selection, "chosen", temperature)

result = {"original_full_input_sha256": record["original_full_input_sha256"], "temperature_selection": selection,
          "chosen_temperature": temperature, "training_reported_steps": v["/results/audio/training/steps"],
          "training_reported_weights_changed": v["/results/audio/training/weights_changed"],
          "training_history_length": len(v["/results/audio/training/history"]), "recomputed": {}}
history = v["/results/audio/training/history"]
assert len(history) == v["/results/audio/training/steps"] == 250
assert sum(row["effective_targets"] for row in history) == v["/results/audio/training/effective_targets"]
assert any(row["grad_norm"] > 0 for row in history) == v["/results/audio/training/nonzero_gradient_seen"]
for split in ["validation", "test"]:
    base = f"/results/audio/calibration/{split}"
    logits = torch.tensor(v[base + "/logits"], dtype=torch.float32)
    labels = torch.tensor(v[base + "/labels"], dtype=torch.long)
    assert len(labels) == len(splits[split]) == v[base + "/count"]
    assert labels.tolist() == [namespace["classes"].index(r["answer"]) for r in splits[split]]
    assert v[base + "/families"] == [r["family"] for r in splits[split]]
    assert torch.equal(logits.argmax(-1), (logits / temperature).argmax(-1))
    result["recomputed"][split] = {}
    for phase, t in [("original", 1.0), ("calibrated", temperature)]:
        observed = namespace["confidence_metrics"](logits, labels, t)
        for field in ["count", "correct", "accuracy", "predicted_labels"]:
            assert observed[field] == v[f"{base}/{phase}/{field}"]
        for field in ["probabilities", "confidence"]:
            expected = torch.tensor(v[f"{base}/{phase}/{field}"])
            assert torch.allclose(torch.tensor(observed[field]), expected, atol=1e-6, rtol=0)
        assert abs(observed["nll"] - v[f"{base}/{phase}/nll"]) <= 1e-6
        result["recomputed"][split][phase] = {k: observed[k] for k in ["count", "correct", "accuracy", "confidence", "predicted_labels", "nll"]}
        print("replayed", split, phase, "count", observed["count"], "correct", observed["correct"], "nll", observed["nll"])

row_index = next(i for i, row in enumerate(splits["test"]) if row["frequency"] == 320 and row["amplitude"] == 0.5 and row["seconds"] == 0.12)
sample = splits["test"][row_index]
original = result["recomputed"]["test"]["original"]
calibrated = result["recomputed"]["test"]["calibrated"]
assert sample["frequency"] not in {r["frequency"] for r in splits["train"]}
assert original["predicted_labels"][row_index] == calibrated["predicted_labels"][row_index] == 0
assert sample["answer"] == "high" and namespace["classes"][0] == "low"
assert round(original["confidence"][row_index], 4) == 0.7490
assert round(calibrated["confidence"][row_index], 4) == 0.8990
result["section_sample"] = {"row": row_index, "record": sample,
                            "logits": v["/results/audio/calibration/test/logits"][row_index],
                            "original_confidence": original["confidence"][row_index],
                            "calibrated_confidence": calibrated["confidence"][row_index],
                            "prediction": "low", "correct": False}
print("section_sample", result["section_sample"])
print("training_report", result["training_reported_steps"], "steps; weights_changed", result["training_reported_weights_changed"])
(ARTIFACT / "execution/encoders-recomputed.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("PASS saved-logit replay only; no training, model load, or model inference")
