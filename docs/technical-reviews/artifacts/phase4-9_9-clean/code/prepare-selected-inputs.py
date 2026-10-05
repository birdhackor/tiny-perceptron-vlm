"""Read only named original measurement/provenance JSON pointers; no annotations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1] / "inputs"
SOURCE = ROOT / "docs/course-experiments/results/encoders.json"
raw = SOURCE.read_bytes()
document = json.loads(raw)
pointers = [
    "/revision", "/device", "/seed", "/torch_version", "/python_version", "/step_scale",
    "/code_sha256/scripts~1course_experiments~1modalities.py",
    "/code_sha256/tiny_perceptron~1multimodal.py",
    "/results/audio/classes", "/results/audio/config",
    "/results/audio/training/steps", "/results/audio/training/weights_changed",
    "/results/audio/training/nonzero_gradient_seen", "/results/audio/training/effective_targets",
    "/results/audio/training/initial_loss", "/results/audio/training/final_loss",
    "/results/audio/training/history",
    "/results/audio/calibration/temperature_grid", "/results/audio/calibration/selection",
    "/results/audio/calibration/chosen_temperature", "/results/audio/calibration/test_argmax_invariant",
]
for split in ("train", "validation", "test"):
    pointers += [f"/results/audio/data/splits/{split}/{field}" for field in ("count", "sha256", "records")]
for split in ("validation", "test"):
    base = f"/results/audio/calibration/{split}"
    pointers += [f"{base}/{field}" for field in ("count", "logits", "labels", "families")]
    for phase in ("original", "calibrated"):
        pointers += [f"{base}/{phase}/{field}" for field in
                     ("temperature", "count", "correct", "accuracy", "probabilities", "confidence", "predicted_labels", "nll")]

selected = {}
for pointer in pointers:
    value = document
    for part in pointer.strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        assert part not in {"notes", "review"} and not part.endswith("_scope_correction")
        value = value[part]
    selected[pointer] = value
receipt = {
    "original_path": str(SOURCE.relative_to(ROOT)),
    "original_full_input_sha256": hashlib.sha256(raw).hexdigest(),
    "inspection_pointers": pointers,
    "policy": "Only these original measurement/provenance pointers inspected; original JSON remains unchanged.",
    "selected_values": selected,
}
(OUT / "encoders-selected-pointers.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
for key in ["/results/audio/classes", "/results/audio/config", "/results/audio/training/steps",
            "/results/audio/training/weights_changed", "/results/audio/calibration/chosen_temperature",
            "/results/audio/calibration/selection"]:
    print(key, selected[key])
for split in ("train", "validation", "test"):
    base = f"/results/audio/data/splits/{split}"
    print(base, "count", selected[base + "/count"], "frequencies", sorted({r["frequency"] for r in selected[base + "/records"]}))
for split in ("validation", "test"):
    base = f"/results/audio/calibration/{split}"
    print(base, "logits", selected[base + "/logits"], "labels", selected[base + "/labels"], "families", selected[base + "/families"])
print("original_full_input_sha256", receipt["original_full_input_sha256"])
