"""Independent CPU replay and full-record audit; never train or modify weights.

Published fixed revision is sufficient when ignored integration weights are absent.
All downloaded files are anonymous (token=False), pinned, and SHA/size checked.
"""

import copy
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.capstone import _balanced_sample  # noqa: E402
from scripts.course_experiments.capstone_deployment import (  # noqa: E402
    counterfactual_pairs,
    image_counterfactuals,
)
from tiny_perceptron.capstone import (  # noqa: E402
    TOK,
    build_dataset,
    digest,
    encode_record,
    evaluate_rows,
    load_capstone,
    modality_tensors,
    preference_pairs,
    prepare_batch,
    prompt_ids,
)
from tiny_perceptron.data import IGNORE  # noqa: E402
from tiny_perceptron.multimodal import log_mel, tone  # noqa: E402

OUT = Path(__file__).parent
PREFIX = "fact_v2_19_06_"
EVIDENCE = ROOT / "docs/course-experiments/capstone-evidence"
ENV = {
    "python": sys.version,
    "torch": torch.__version__,
    "torch_git": torch.version.git_version,
    "huggingface_hub": __import__("huggingface_hub").__version__,
    "device": "cpu",
    "dtype": "float32",
    "platform": platform.platform(),
    "threads": "2",
}
torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / (PREFIX + name + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def tensor_inventory(model):
    result = {}
    for name, tensor in model.state_dict().items():
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "elements": tensor.numel(),
            "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
            "finite": bool(tensor.isfinite().all()),
        }
    assert all(item["finite"] for item in result.values())
    return result


manifest = read(ROOT / "docs/course-experiments/capstone-public.json")
assert manifest["revision"] == "33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed"
private_paths = {
    "sft": "outputs/integration-runs/v2-sft/capstone-review/capstone_sft/gha-37168151999-1/model.pt",
    "joint": "outputs/integration-runs/v2-joint/capstone-review/capstone_joint/gha-37168518451-1/model.pt",
    "dpo": "outputs/integration-runs/v2-dpo/capstone-review/capstone_preference/gha-37168874504-1/model.pt",
}
models, checkpoint_checks = {}, {}
for stage, local_name in private_paths.items():
    specification = next(row for row in manifest["models"] if row["id"] == stage)
    item = next(row for row in specification["files"] if row["output"] == "model.pt")
    public_file = Path(hf_hub_download(manifest["repo"], item["path"], revision=manifest["revision"], token=False))
    assert sha(public_file) == item["sha256"] and public_file.stat().st_size == item["bytes"]
    public_model, public_payload = load_capstone(public_file)
    assert public_payload["stage"] == stage and public_payload["inference_only"] is True
    assert public_payload["tokenizer"] == TOK.state()
    assert public_payload["step"] == {"sft": 1400, "joint": 600, "dpo": 100}[stage]
    public_inventory = tensor_inventory(public_model)
    local_file = ROOT / local_name
    local_details = None
    if local_file.exists():
        local_model, local_payload = load_capstone(local_file)
        assert local_payload["metadata"]["schedule_completed"] is True
        local_inventory = tensor_inventory(local_model)
        assert local_inventory == public_inventory
        assert public_payload["metadata"]["source_checkpoint_sha256"] == sha(local_file)
        local_details = {"path": local_name, "sha256": sha(local_file), "bytes": local_file.stat().st_size,
                         "payload_except_model": {key: value for key, value in local_payload.items() if key != "model"},
                         "all_public_tensors_bitwise_equal": True}
        models[stage] = local_model
    else:
        models[stage] = public_model
    checkpoint_checks[stage] = {
        "public": {"repo": manifest["repo"], "revision": manifest["revision"], "token": False,
                   "path": item["path"], "sha256": sha(public_file), "bytes": public_file.stat().st_size},
        "public_payload_except_model": {key: value for key, value in public_payload.items() if key != "model"},
        "local": local_details, "tensor_inventory": public_inventory,
        "model_description": public_model.description(),
    }
save("checkpoints", {"environment": ENV, "checkpoints": checkpoint_checks})

splits, data_manifest = build_dataset()
stored_data = read(EVIDENCE / "deployment/data.json")
assert stored_data["manifest"] == data_manifest and stored_data["splits"] == splits
save("dataset_audit", {"manifest": data_manifest, "full_data_equals_rebuilt": True,
    "split_hashes": {name: digest(rows) for name, rows in splits.items()},
    "data_file_sha256": sha(EVIDENCE / "deployment/data.json"),
    "all_rows": [{"split": split, "id": row["id"], "family": row["family"], "task": row["task"],
        "system": row["system"], "user": row["user"], "answer": row["answer"], "image": row["image"],
        "audio": row["audio"], "prompt_ids": prompt_ids(row)} for split, rows in splits.items() for row in rows]})

# Execute the exact lesson block, preserving its teacher-forced inputs and masks.
section = (OUT / (PREFIX + "read_19.6.md")).read_text()
code = section.split("```python\n", 1)[1].split("```", 1)[0]
namespace = {}
torch.manual_seed(42)  # The printed shape invariants do not depend on initialization.
exec(compile(code, "course/chapters/19.md#19.6", "exec"), namespace)
row = namespace["row"]
batch, labels = namespace["batch"], namespace["labels"]
result = namespace["result"]
image, audio = namespace["image"], namespace["audio"]
prefix = prompt_ids(row)
answer_ids = TOK.encode(row["answer"]) + [TOK.eos_id]
start = len(prefix) - 1
assert labels[0, start:].tolist() == answer_ids
assert (labels[0, :start] == IGNORE).all()
assert batch["ids"][0, start].item() == TOK.assistant_id
assert batch["ids"].dtype == torch.long and labels.dtype == torch.long and batch["valid"].dtype == torch.bool
assert not result["logits"].requires_grad
assert result["logits"].isfinite().all()
pooled = torch.nn.functional.adaptive_avg_pool2d(image.unsqueeze(0), (4, 4))
manual = image.reshape(3, 4, 4, 4, 4).mean((2, 4))
assert torch.allclose(pooled[0], manual, atol=1e-6, rtol=0)
audio_spec = row["audio"]
frequency = (440 if audio_spec["pitch"] == "low" else 880) + audio_spec["delta"] + (audio_spec["variation"] - 1) * 2
mel = log_mel(tone(frequency, seconds=0.04), bands=16)
assert torch.equal(mel.mean(-1), audio)
assert torch.allclose(mel.mean(-1), mel.flip(-1).mean(-1), atol=1e-6, rtol=0)
exercise = next(row for row in splits["train"] if row["task"] == "image_color")
exercise_image, exercise_audio = modality_tensors(exercise)
exercise_batch, exercise_labels = prepare_batch([exercise])
assert exercise_audio is None and TOK.audio_id not in prompt_ids(exercise)
with torch.no_grad():
    exercise_scores = namespace["model"](**exercise_batch)["logits"]
assert exercise_scores.shape[:2] == exercise_labels.shape
marker_failure = None
try:
    namespace["model"](torch.tensor([[TOK.audio_id]]))
except ValueError as error:
    marker_failure = str(error)
assert marker_failure
save("snippet", {"environment": ENV, "source_sha256": hashlib.sha256(section.encode()).hexdigest(),
    "literal_code": code, "row": row, "image_shape": list(image.shape), "image_dtype": str(image.dtype),
    "audio_shape": list(audio.shape), "audio_dtype": str(audio.dtype), "frequency_hz": frequency,
    "waveform_samples": 640, "mel_shape": list(mel.shape), "mean_permutation_invariance": True,
    "pool_manual_max_abs_error": float((pooled[0] - manual).abs().max()),
    "pooled_shape": list(pooled.shape), "flattened_shape": list(pooled.flatten(1).shape),
    "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
    "ids": batch["ids"].tolist(), "valid": batch["valid"].tolist(), "labels": labels.tolist(),
    "logits_shape": list(result["logits"].shape), "effective_targets": int((labels != IGNORE).sum()),
    "first_answer_target_index": start, "grad_disabled": not result["logits"].requires_grad,
    "projector_shapes": {"image": list(namespace["model"].image_projector[0].weight.shape),
                         "audio": list(namespace["model"].audio_projector[0].weight.shape)},
    "exercise_audio_is_none": exercise_audio is None, "exercise_has_no_audio_marker": True,
    "exercise_logits_shape": list(exercise_scores.shape), "missing_marker_features_error": marker_failure})

# Reconstruct effective targets from the exact sampler sequence, with no forward/backward or updates.
schedules = {}
for stage, steps in [("sft", 1400), ("joint", 600), ("dpo", 100)]:
    index = ["pretrain", "sft", "joint", "dpo"].index(stage)
    sampler = random.Random(42 + index * 1000)
    training_rows = [row for row in splits["train"] if row["image"] is None and row["audio"] is None] if stage == "sft" else splits["train"]
    lengths = {row["id"]: int((encode_record(row)[1] != IGNORE).sum()) for row in training_rows}
    counts = []
    for _ in range(steps):
        sampled = _balanced_sample(training_rows, sampler, 24)
        counts.append(sum(lengths[row["id"]] for row in sampled))
        if stage == "dpo":
            sampler.choices(preference_pairs(splits["train"]), k=12)
    experiment = "capstone_preference" if stage == "dpo" else f"capstone_{stage}"
    formal = read(ROOT / f"docs/course-experiments/results/{experiment}.json")
    assert sum(counts) == formal["results"]["effective_tokens"]
    assert formal["results"]["steps"] == steps and formal["results"]["test_evaluated"] is False
    for path in ["tiny_perceptron/capstone.py", "scripts/course_experiments/capstone.py", "tiny_perceptron/multimodal.py"]:
        assert sha(ROOT / path) == formal["code_sha256"][path]
    schedules[stage] = {"per_step_effective_targets": counts, "sum": sum(counts), "updates": steps,
        "batch_size": 24, "seed": 42, "sampler_seed": 42 + index * 1000,
        "lr": {"sft": .003, "joint": .0015, "dpo": .0002}[stage], "router_balance_coefficient": .01,
        "optimizer": "AdamW(default betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)",
        "gradient_clip_norm": 1.0, "dpo_beta": .1 if stage == "dpo" else None,
        "dpo_replay_ce_coefficient": .2 if stage == "dpo" else None,
        "preference_pairs_per_dpo_update": 12 if stage == "dpo" else None,
        "formal_result": str((ROOT / f"docs/course-experiments/results/{experiment}.json").relative_to(ROOT)),
        "formal_result_sha256": sha(ROOT / f"docs/course-experiments/results/{experiment}.json"),
        "formal_configuration": {key: value for key, value in formal.items() if key != "results"},
        "formal_training_report": formal["results"],
        "audit_scope": "No training repeated; exact sampler target denominators reconstructed. Historical GPU wall times are recorded, not CPU-remeasured."}
save("schedule_audit", {"environment": ENV, "schedules": schedules, "optimizer_updates_in_this_audit": 0})

image_swaps = image_counterfactuals(splits["test"])
audio_swaps = []
for original in splits["test"]:
    if original["task"] != "joint":
        continue
    swapped = copy.deepcopy(original)
    swapped["audio"]["pitch"] = "high" if original["audio"]["pitch"] == "low" else "low"
    swapped["answer"] = f"DIRECT:{original['image']['color']},{swapped['audio']['pitch']}"
    swapped["id"] = original["id"] + "-audio-swap"
    audio_swaps.append(swapped)
swaps_details = []
originals_by_id = {row["id"]: row for row in splits["test"]}
for rows, suffix in [(image_swaps, "-image-swap"), (audio_swaps, "-audio-swap")]:
    for swapped in rows:
        original = originals_by_id[swapped["id"].removesuffix(suffix)]
        assert swapped["user"] == original["user"] and prompt_ids(swapped) == prompt_ids(original)
        before_image, before_audio = modality_tensors(original)
        after_image, after_audio = modality_tensors(swapped)
        if suffix == "-image-swap":
            assert not torch.equal(before_image, after_image)
            if before_audio is not None:
                assert torch.equal(before_audio, after_audio)
        else:
            assert torch.equal(before_image, after_image) and not torch.equal(before_audio, after_audio)
        swaps_details.append({"original": original, "swapped": swapped,
            "same_prompt_ids": True, "image_equal": torch.equal(before_image, after_image),
            "audio_equal": None if before_audio is None else torch.equal(before_audio, after_audio)})
save("swap_inputs", swaps_details)


def verify_records(stored, rows):
    by_id = {row["id"]: row for row in rows}
    assert len(by_id) == len(stored["records"]) == stored["count"]
    summary, inspected = {}, []
    for record in stored["records"]:
        row = by_id[record["id"]]
        assert record["expected_action"] == row["answer"]
        assert record["action_trace"]["prompt_ids"] == prompt_ids(row)
        for name in ["action_trace", "final_trace"]:
            trace = record[name]
            if trace is None:
                continue
            assert TOK.decode(trace["generated_ids"]) == trace["raw"]
            assert trace["eos"] == bool(trace["generated_ids"] and trace["generated_ids"][-1] == TOK.eos_id)
        action_correct = record["action_trace"]["eos"] and record["action_trace"]["raw"] == row["answer"]
        final_expected = record["expected_final"]
        if row["task"] == "calculator":
            parameters = row["answer"].rsplit(":", 1)[1].split("+")
            assert final_expected == str(sum(map(int, parameters)))
        else:
            assert final_expected == row["answer"].split(":", 1)[1]
        end_correct = action_correct and record["answer"] == final_expected
        assert record["action_correct"] == action_correct and record["end_to_end_correct"] == end_correct
        total = summary.setdefault(row["task"], {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
        total["count"] += 1
        total["action_correct"] += int(action_correct)
        total["end_to_end_correct"] += int(end_correct)
        inspected.append({"id": record["id"], "task": row["task"], "image": row["image"], "audio": row["audio"],
            "expected": row["answer"], "raw": record["action_trace"]["raw"], "eos": record["action_trace"]["eos"],
            "end_correct": end_correct})
    assert summary == stored["by_task"]
    assert sum(row["action_correct"] for row in summary.values()) == stored["action_correct"]
    assert sum(row["end_to_end_correct"] for row in summary.values()) == stored["end_to_end_correct"]
    return summary, inspected


jobs = [("sft_validation", "sft", splits["validation"], "sft/validation.json"),
        ("joint_validation", "joint", splits["validation"], "joint/validation.json"),
        ("dpo_validation", "dpo", splits["validation"], "dpo/validation.json"),
        ("joint_test", "joint", splits["test"], "deployment/test-joint.json"),
        ("dpo_test", "dpo", splits["test"], "deployment/test-dpo.json"),
        ("joint_image_swaps", "joint", image_swaps, "deployment/test-joint-image-swaps.json"),
        ("joint_audio_swaps", "joint", audio_swaps, "deployment/test-joint-audio-swaps.json")]
audits, replays = {}, {}
for name, stage, rows, evidence_path in jobs:
    stored = read(EVIDENCE / evidence_path)
    summary, inspected = verify_records(stored, rows)
    started = time.perf_counter()
    replay = evaluate_rows(models[stage], rows, max_new_tokens=64, batch_size=24)
    elapsed = time.perf_counter() - started
    verify_records(replay, rows)
    old = {record["id"]: record for record in stored["records"]}
    differences = [{"id": record["id"], "expected": old[record["id"]], "actual": record}
                   for record in replay["records"] if record != old[record["id"]]]
    replay["audit_metadata"] = {"environment": ENV, "seconds": elapsed, "command":
        ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_06_replay.py",
        "formal_source": evidence_path, "formal_source_sha256": sha(EVIDENCE / evidence_path),
        "stage": stage, "max_new_tokens": 64, "batch_size": 24, "generation": "greedy+KV cache",
        "differences": differences}
    save(name + "_replay", replay)
    audits[name] = {"by_task": summary, "all_records_inspected": inspected, "record_differences": len(differences),
        "replay_seconds": elapsed, "source_sha256": sha(EVIDENCE / evidence_path)}
    replays[name] = replay
    print(name, "records", len(rows), "differences", len(differences), "seconds", round(elapsed, 3), flush=True)

pairs = counterfactual_pairs(replays["joint_test"], replays["joint_image_swaps"])
stored_pairs = read(EVIDENCE / "deployment/test-joint-image-pairs.json")
assert {key: value for key, value in stored_pairs.items() if key != "diagnostic_stage"} == pairs
pair_by_task = {}
for pair in pairs["pairs"]:
    task = originals_by_id[pair["original_id"]]["task"]
    count = pair_by_task.setdefault(task, {"count": 0, "both_correct": 0, "answer_changed": 0})
    count["count"] += 1
    count["both_correct"] += int(pair["both_end_to_end_correct"])
    count["answer_changed"] += int(pair["generated_answer_changed"])
base_by_id = {record["id"]: record for record in replays["joint_test"]["records"]}
audio_both = sum(record["end_to_end_correct"] and base_by_id[record["id"].removesuffix("-audio-swap")]["end_to_end_correct"]
                 for record in replays["joint_audio_swaps"]["records"])
joint_val = {record["id"]: record for record in replays["joint_validation"]["records"]}
regressions = [{"joint": joint_val[record["id"]], "dpo": record} for record in replays["dpo_validation"]["records"]
    if joint_val[record["id"]]["end_to_end_correct"] and not record["end_to_end_correct"]]
assert len(regressions) == 4
save("audit", {"environment": ENV, "code_hashes": {path: sha(ROOT / path) for path in [
    "tiny_perceptron/capstone.py", "tiny_perceptron/multimodal.py", "tiny_perceptron/data.py",
    "tiny_perceptron/model.py", "scripts/course_experiments/capstone.py",
    "scripts/course_experiments/capstone_deployment.py", "scripts/course_experiments/modalities.py"]},
    "audits": audits, "image_pairs": pairs, "image_pair_by_task": pair_by_task,
    "audio_both_correct": audio_both, "dpo_regressions": regressions,
    "constant_audio_guess": {split: dict(Counter(row["answer"] for row in rows if row["task"] == "audio"))
                             for split, rows in splits.items()},
    "formal_deployment_result_sha256": sha(ROOT / "docs/course-experiments/results/capstone_deployment.json"),
    "formal_deployment_configuration": read(ROOT / "docs/course-experiments/results/capstone_deployment.json"),
    "selection_sha256": sha(ROOT / "docs/course-experiments/capstone-selection.json"),
    "selection": read(ROOT / "docs/course-experiments/capstone-selection.json"),
    "optimizer_updates_in_this_audit": 0})
assert all(audit["record_differences"] == 0 for audit in audits.values())
print("All full records exactly replayed; no optimizer updates. Image pairs:", pair_by_task, "Audio pairs:", audio_both)
