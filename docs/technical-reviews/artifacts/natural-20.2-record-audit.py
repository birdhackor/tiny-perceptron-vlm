from pathlib import Path
import hashlib
import json
import platform

root = Path(__file__).resolve().parents[3]
base = root / "docs/natural-assistant/evidence"
prepare = json.loads((base / "prepare/result.json").read_text())
baseline = json.loads((base / "baseline/result.json").read_text())
train = json.loads((base / "train/result.json").read_text())
cpu = json.loads((base / "usage/cpu-base-input-routes.json").read_text())
text = json.loads((base / "usage/cpu-base-text.json").read_text())
def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()
fingerprints = []
for key, name in [("core", "Qwen--Qwen3-VL-2B-Instruct"), ("asr", "openai--whisper-small")]:
    snap = prepare["snapshots"][key]
    for filename in ("README.md", "config.json", "model.safetensors"):
        receipt = next(row for row in snap["files"] if row["name"] == filename)
        path = root / "outputs/natural-extension/student-base-cache/hf" / ("models--" + name) / "snapshots" / snap["revision"] / filename
        actual = digest(path)
        assert actual == receipt["sha256"], (filename, actual)
        assert path.stat().st_size == receipt["bytes"]
        if filename != "model.safetensors":
            own = root / "docs/technical-reviews/artifacts/natural-20.2-sources" / (("qwen-" if key == "core" else "whisper-") + ("card.md" if filename == "README.md" else "config.json"))
            assert digest(own) == actual
        fingerprints.append({"model": name, "revision": snap["revision"], "name": filename, "bytes": receipt["bytes"], "sha256": actual})
assert baseline["status"] == "completed"
assert baseline["total_parameters"] == 2127532032
assert baseline["trainable_parameters"] == 0
assert baseline["gpu_name"] == "NVIDIA L4"
assert baseline["device"] == "cuda"
assert baseline["execution"]["adapter_run_id"] is None
assert cpu["parameters"] == {"total_parameters": 2127532032, "trainable_parameters": 0}
assert cpu["provenance"]["device"] == "cpu"
assert cpu["provenance"]["dtype"] == "float32"
assert len(cpu["image_generations"]) == 2
assert cpu["audio"]["audio_seconds"] == 7.8
images = [{key: row.get(key) for key in ("id", "split", "task", "reference_answer", "prediction", "input_tokens", "generated_tokens", "score")} for row in cpu["image_generations"]]
ocr = next(row for row in images if row["task"] == "ocr")
assert ocr["reference_answer"] == "臺灣學生一起讀書。"
assert ocr["prediction"] == "台灣學生一起讀書。"
assert ocr["score"]["errors"] == 1 and ocr["score"]["reference_characters"] == 9
assert train["trainable_parameters"] == 1605632
assert train["optimizer_only_lora"] is True
assert set(train["optimizer_parameter_names"]) == set(train["trainable_parameter_names"])
assert len(train["trainable_parameter_names"]) == 112
assert all("lora_" in name for name in train["optimizer_parameter_names"])
assert train["total_parameters"] == 2127532032 + 1605632
assert train["completed_steps"] == 180
assert train["changed_adapter_tensor_count"] == 112
assert train["frozen_parameter_samples_unchanged"] is True
result = {
    "reviewer_task": "/root/natural_factual_20_2", "environment": {"python": platform.python_version(), "device": "cpu, file inspection only"},
    "fixed_snapshot_fingerprints": fingerprints,
    "baseline": {key: baseline[key] for key in ("status", "total_parameters", "trainable_parameters", "gpu_name", "device", "dtype", "versions", "split", "seed", "requested_visual_text_rows", "requested_audio_rows", "execution")},
    "cpu": {"provenance": {k: cpu["provenance"][k] for k in ("model_revision", "asr_revision", "versions", "device", "dtype", "seed", "max_pixels")}, "parameters": cpu["parameters"], "images": images, "audio_seconds": cpu["audio"]["audio_seconds"], "transcript": cpu["audio"]["transcript"], "speech_prediction": cpu["speech_response"]["prediction"], "speech_generated_tokens": cpu["speech_response"]["generated_tokens"], "text_prompt": text["user"], "text_prediction": text["prediction"], "text_generated_tokens": text["generated_tokens"]},
    "train": {key: train[key] for key in ("total_parameters", "trainable_parameters", "lora_rank", "lora_targets", "completed_steps", "optimizer_only_lora", "changed_adapter_tensor_count", "frozen_parameter_samples_unchanged", "frozen_check_scope")},
    "scope": "Independent inspection of completed upstream run records and model file SHA verification. Existing CPU/GPU runs were not repeated; CPU photo exact mismatch is not a semantic quality grade. No heldout adapter quality or unfinished chapter result is endorsed.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
