"""Reaggregate existing raw audio result JSON, without loading/evaluating a model."""
import hashlib
import json
import platform
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
PATH = ROOT / "docs/course-experiments/results/audio.json"
raw = PATH.read_bytes()
data = json.loads(raw)
test = data["results"]["test"]
splits = data["results"]["data"]["splits"]
train = data["results"]["training"]
encode = lambda text: [x + 8 for x in text.encode("utf-8")]
checks = []
failed = []
for row, sample in zip(splits["test"]["records"], test["samples"], strict=True):
    expected = "high" if row["frequency"] > 300 else "low"
    assert sample["target"] == row["answer"] == expected
    ids = sample["generated_ids"]
    answer_ids = ids[:ids.index(2)] if 2 in ids else ids
    match = answer_ids == encode(expected)
    assert sample["exact_match"] == match
    assert sample["eos"] == (2 in ids)
    assert sample["invalid_special_tokens"] == sum(x < 8 for x in answer_ids)
    decoded = bytes(x - 8 for x in answer_ids if x >= 8).decode("utf-8", errors="replace")
    assert decoded == sample["generated"]
    inspected = {"row": sample["row"], "frequency_hz": row["frequency"],
        "amplitude": row["amplitude"], "seconds": row["seconds"], "target": expected,
        "generated": decoded, "raw_generated_ids": ids, "exact_match_recomputed": match, "eos": 2 in ids}
    checks.append(inspected)
    if not match:
        failed.append(inspected)
correct = sum(x["exact_match_recomputed"] for x in checks)
effective = sum(len(encode(row["answer"])) + 1 for row in splits["test"]["records"])
assert correct == test["correct"] == 11
assert len(checks) == test["examples"] == splits["test"]["count"] == 14
assert effective == test["effective_tokens"] == 62
assert test["exact_match"] == correct / len(checks)
assert sum(x["eos"] for x in checks) / len(checks) == test["eos_rate"] == 1.0
assert test["skipped"] == []
assert test["generation_errors"] == 0
assert sum(x["generation_error"] is not None for x in test["samples"]) == 0
frequencies = {name: sorted({r["frequency"] for r in split["records"]}) for name, split in splits.items()}
for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    assert not set(frequencies[a]) & set(frequencies[b])
for name, split in splits.items():
    assert split["count"] == len(split["records"])
    for r in split["records"]:
        assert r["answer"] == ("high" if r["frequency"] > 300 else "low")
        assert (r["amplitude"], r["seconds"]) in {(0.25, 0.1), (0.5, 0.12)}
assert len(train["history"]) == train["steps"] == 300
assert [r["step"] for r in train["history"]] == list(range(1, 301))
assert sum(r["effective_targets"] for r in train["history"]) == train["effective_targets"] == train["effective_tokens"] == 5409
code_checks = []
for name in ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "tiny_perceptron/multimodal.py",
             "tiny_perceptron/attention.py", "scripts/course_experiments/modalities.py"]:
    current = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    recorded = data["code_sha256"][name]
    assert current == recorded
    code_checks.append({"path": name, "sha256": current, "matches_recorded_run_code_sha256": True})
pointers = ["/experiment_id", "/revision", "/device", "/seed", "/torch_version", "/python_version",
    "/step_scale", "/artifacts", "/code_sha256",
    "/results/data/splits/train", "/results/data/splits/validation", "/results/data/splits/test",
    "/results/training/parameters", "/results/training/trainable_parameters", "/results/training/config",
    "/results/training/modal_config", "/results/training/initial_loss", "/results/training/final_loss",
    "/results/training/history", "/results/training/steps", "/results/training/effective_targets",
    "/results/training/effective_tokens", "/results/training/weights_changed", "/results/training/nonzero_gradient_seen",
    "/results/training/cpu_smoke", "/results/training/checkpoint",
    "/results/test/examples", "/results/test/correct", "/results/test/exact_match", "/results/test/effective_tokens",
    "/results/test/eos_rate", "/results/test/generation_errors", "/results/test/invalid_special_tokens",
    "/results/test/ablation", "/results/test/skipped", "/results/test/groups", "/results/test/macro_accuracy",
    "/results/test/samples"]
report = {"method": "Existing raw JSON reaggregation; no model loaded, trained or re-evaluated",
    "raw_path": PATH.relative_to(ROOT).as_posix(), "raw_sha256": hashlib.sha256(raw).hexdigest(),
    "inspected_json_pointers": pointers, "environment": {"python": platform.python_version(), "device": "CPU, JSON-only"},
    "provenance": {k:data[k] for k in ["revision", "device", "seed", "torch_version", "python_version", "step_scale"]},
    "code_identity_checks": code_checks, "split_counts": {k:v["count"] for k,v in splits.items()},
    "split_frequency_hz": frequencies, "test_correct": correct, "test_examples": len(checks),
    "test_exact_match": correct / len(checks), "test_target_tokens_including_eos": effective,
    "eos_correctly_ended_count": sum(x["eos"] for x in checks), "training_steps": train["steps"],
    "training_effective_targets": train["effective_targets"], "training_weights_changed_recorded": train["weights_changed"],
    "samples_recomputed": checks, "failed_rows": failed,
    "amplitude_duration_pairs": [[0.25, 0.1], [0.5, 0.12]],
    "independent_inference": "All three errors are at 290/300 Hz. Duration and amplitude are paired, both durations occur in every split, and neither factor has an isolated control. These records cannot isolate a duration effect."}
(ART / "raw-reaggregation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
