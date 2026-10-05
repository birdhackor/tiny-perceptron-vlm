"""Bounded CPU data-integrity verification; no model loading or training."""
import contextlib
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

from tiny_perceptron.capstone import build_dataset, digest, modality_tensors, prompt_ids

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def actual_input_sha(row):
    value = hashlib.sha256(json.dumps(prompt_ids(row)).encode())
    for feature in modality_tensors(row):
        if feature is not None:
            assert feature.device.type == "cpu"
            value.update(feature.numpy().tobytes())
    return value.hexdigest()

def emit(label, value):
    print(label, json.dumps(value, ensure_ascii=False, sort_keys=True))

emit("environment", {"python": sys.version, "torch": str(torch.__version__),
                     "torch_git_version": str(torch.version.git_version), "device": "cpu",
                     "cuda_build": str(torch.version.cuda), "threads": torch.get_num_threads()})
with contextlib.redirect_stdout(io.StringIO()) as captured:
    original = (HERE / "original-fence.py").read_bytes()
    exec(compile(original, "course/chapters/19.md#19.3:original-fence", "exec"), {})
print("ORIGINAL_FENCE_BEGIN")
print(captured.getvalue(), end="")
print("ORIGINAL_FENCE_END")

frozen_raw = (HERE / "sources/data-1df3353.json").read_bytes()
assert frozen_raw == (ROOT / "docs/course-experiments/capstone-evidence/deployment/data.json").read_bytes()
frozen = json.loads(frozen_raw)
emit("frozen_json_top_keys_types", {k: type(v).__name__ for k, v in frozen.items()})
emit("manifest_keys_types", {k: type(v).__name__ for k, v in frozen["manifest"].items()})
splits, manifest = build_dataset(seed=42)
assert splits == frozen["splits"] and manifest == frozen["manifest"]
assert manifest["version"] == "capstone-small-world-v2"
counts = {name: len(rows) for name, rows in splits.items()}
assert counts == {"train": 552, "validation": 84, "test": 90}
assert sum(counts.values()) == 726
emit("counts", counts)
computed = {name: digest(rows) for name, rows in splits.items()}
assert computed == manifest["sha256"]
assert len(set(computed.values())) == 3
emit("computed_split_sha256", computed)

for stage in ["pretrain", "sft", "joint", "dpo"]:
    saved = json.loads((HERE / f"sources/{stage}-data-manifest.json").read_bytes())
    assert saved["sha256"] == computed and saved["counts"] == counts
    assert saved["seed"] == 42 and saved["version"] == manifest["version"]
student = json.loads((HERE / "sources/student-report-original.json").read_bytes())
assert student["data_manifest"]["sha256"] == computed
emit("frozen_stage_provenance", {"stages": ["pretrain", "sft", "joint", "dpo"],
                                "comparison": "student /data_manifest", "split_sha256_equal": True})

families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
inputs = {name: {actual_input_sha(row) for row in rows} for name, rows in splits.items()}
overlaps = {}
for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    overlaps[f"{left}/{right}"] = {"families": len(families[left] & families[right]),
                                   "actual_inputs": len(inputs[left] & inputs[right])}
    assert overlaps[f"{left}/{right}"] == {"families": 0, "actual_inputs": 0}
emit("cross_split_overlaps", overlaps)
for name, rows in splits.items():
    family_names = set()
    for row in rows:
        family_names.add(row["family"])
        assert row["id"] == digest({k: v for k, v in row.items() if k != "id"})[:20]
    assert len({row["id"] for row in rows}) == len(rows)
    emit(f"{name}_task_counts", dict(Counter(row["task"] for row in rows)))

audio_summary = {}
for name, nfamily in [("train", 8), ("validation", 1), ("test", 1)]:
    rows = [row for row in splits[name] if row["task"] == "audio"]
    labels = Counter(row["audio"]["pitch"] for row in rows)
    family_count = Counter(f.split(":")[1] for f in {row["family"] for row in rows})
    assert labels == {"high": nfamily * 3, "low": nfamily * 3}
    assert family_count == {"high": nfamily, "low": nfamily}
    assert all(len([r for r in rows if r["family"] == f]) == 3 for f in {r["family"] for r in rows})
    audio_summary[name] = {"family_counts": dict(family_count), "label_counts": dict(labels),
                           "fixed_high_correct": labels["high"], "denominator": len(rows)}
emit("audio_summary", audio_summary)

image_summary = {}
for name in splits:
    rows = [r for r in splits[name] if r["image"] is not None]
    image_summary[name] = {"colors": sorted({r["image"]["color"] for r in rows}),
                           "shapes": sorted({r["image"]["shape"] for r in rows}),
                           "families": sorted({r["family"] for r in rows})}
assert image_summary["train"]["colors"] == ["blue", "green", "red"]
assert image_summary["train"]["shapes"] == ["circle", "square"]
assert image_summary["validation"]["families"] == ["modalities:blue:circle"]
assert image_summary["test"]["families"] == ["modalities:green:square"]
emit("image_family_summary", image_summary)
for task, label in [("image_color", "green"), ("image_shape", "square")]:
    rows = [r for r in splits["test"] if r["task"] == task]
    correct = sum(r["answer"] == f"DIRECT:{label}" for r in rows)
    assert correct == len(rows) == 9
    assert manifest["task_majority_baselines"]["test"][task] == {
        "count": 9, "correct": 9, "accuracy": 1.0, "majority_label": label}
    emit(task + "_fixed_baseline", {"label": label, "correct": correct, "denominator": len(rows)})

joint = [r for r in splits["test"] if r["task"] == "joint"]
assert len(joint) == 18 and Counter(r["audio"]["pitch"] for r in joint) == {"low": 9, "high": 9}
assert len({r["user"] for r in joint}) == 1
emit("joint_test", {"count": len(joint), "pitch_counts": dict(Counter(r["audio"]["pitch"] for r in joint)),
                    "questions": sorted({r["user"] for r in joint})})
held = [r for r in splits["test"] if r["family"] == "numbers:1:2"]
assert len(held) == 6 and all("numbers:1:2" not in families[name] for name in ["train", "validation"])
assert Counter(r["task"] for r in held) == {"calculator": 2, "unavailable": 2, "concept": 1, "tool_return": 1}
emit("held_1_2", [{k: r[k] for k in ["id", "family", "task", "user", "answer"]} for r in held])
templates = {name: {re.sub(r"\d+", "<n>", r["user"]) for r in rows if r["task"] == "calculator"}
             for name, rows in splits.items()}
assert templates["train"] == templates["validation"] == templates["test"] == {"<n>+<n>等於多少？", "請算<n>加<n>。"}
emit("calculator_templates", {name: sorted(v) for name, v in templates.items()})

# Small counterexamples test the mechanism, never fit a model or create data files.
copied_row = splits["train"][0]
assert len(families["train"] & (families["validation"] | {copied_row["family"]})) == 1
assert len(inputs["train"] & (inputs["validation"] | {actual_input_sha(copied_row)})) == 1
only_high_labels = [r["audio"]["pitch"] for r in splits["test"] if r["task"] == "audio" and r["audio"]["pitch"] == "high"]
assert sum(label == "high" for label in only_high_labels) == len(only_high_labels) == 3
emit("negative_controls", {"copied_train_row_into_validation": {"family_overlap": 1, "input_overlap": 1},
                           "filter_test_audio_to_high": {"fixed_high_correct": 3, "denominator": 3}})
for seed in [0, 7, 99]:
    varied, _ = build_dataset(seed=seed)
    assert Counter(r["audio"]["pitch"] for r in varied["test"] if r["task"] == "audio") == {"low": 3, "high": 3}
    assert len([r for r in varied["test"] if r["family"] == "numbers:1:2"]) == 6
emit("small_seed_variants", {"seeds": [0, 7, 99], "balanced_test_audio_and_reserved_1_2": True})

card = (HERE / "sources/minds14-readme-40ce77c.md").read_text()
frontmatter = card.split("---", 2)[1]
feature_blocks = re.findall(r"(?ms)^  features:\n(.*?)(?=^  splits:)", frontmatter)
fields = [name for block in feature_blocks for name in re.findall(r"(?m)^  - name: ([^\n]+)$", block)]
assert len(feature_blocks) == 15
assert len(fields) == 6 * 15
assert set(fields) == {"path", "audio", "transcription", "english_transcription", "intent_class", "lang_id"}
assert not any("speaker" in field or "session" in field for field in fields)
emit("minds14_schema", {"feature_sets": 15, "fields": sorted(set(fields)), "speaker_id_present": False,
                        "example_path": "fr-FR~ADDRESS/response_4.wav"})
print("ALL_ASSERTIONS_PASSED; only data integrity and constant baselines verified; no model scores")
