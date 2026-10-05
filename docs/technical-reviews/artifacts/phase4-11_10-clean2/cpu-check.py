"""Bounded CPU arithmetic/API checks and read-only verification of original measurements."""
import ast
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.attention import attention_mask, manual_attention
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import VisionEncoder, patchify

torch.set_num_threads(1)
torch.manual_seed(11)
assert not torch.cuda.is_available() and torch.version.cuda is None
HERE = Path(__file__).resolve().parent
result = {
    "environment": {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version), "device": "cpu",
        "cuda_build": str(torch.version.cuda), "cpu_threads": str(torch.get_num_threads()),
    },
    "scope": "No training, backward, optimizer, checkpoint load/save, data download, or existing-model evaluation. Original JSON recomputation is read-only.",
}

arithmetic = []
for height in [16, 32]:
    for patch in [8, 4, 2]:
        visual = (height // patch) ** 2
        total = 20 + visual
        actual = patchify(torch.zeros(1, 3, height, height), patch)
        assert actual.shape == (1, visual, 3 * patch * patch)
        arithmetic.append({"height": height, "patch": patch, "visual": visual,
                           "total": total, "attention_cells": total**2,
                           "patch_tensor_shape": list(actual.shape)})
assert [x["attention_cells"] for x in arithmetic[:3]] == [576, 1296, 7056]
assert arithmetic[-1]["attention_cells"] == 76176
result["arithmetic"] = arithmetic

q = torch.zeros(1, 1, 84, 8)
positions = torch.arange(84)
allowed = attention_mask(positions, positions)
output, weights = manual_attention(q, q, q, allowed)
assert weights.shape == (1, 1, 84, 84) and weights.numel() == 7056
assert int(allowed.sum()) == 84 * 85 // 2 == 3570
result["attention"] = {"weights_shape": list(weights.shape), "full_cells": weights.numel(),
                       "causally_allowed_cells": int(allowed.sum()),
                       "interpretation": "The full score/weight table is N by N; future entries are masked and do not imply access to future answers. This does not benchmark elapsed seconds."}

language = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=64))
with torch.no_grad():
    assert language(embeddings=torch.zeros(1, 64, 8))["logits"].shape == (1, 64, 264)
    try:
        language(embeddings=torch.zeros(1, 84, 8))
    except ValueError as e:
        result["length_limit"] = {"accepted": 64, "rejected": 84, "message": str(e)}
    else:
        raise AssertionError("84 must exceed the configured limit of 64")

image = torch.zeros(1, 3, 16, 16)
entrance = []
for patch in [4, 8, 2]:
    encoder = VisionEncoder(width=16, patch_size=patch)
    entrance.append({"patch": patch, "projection_weight_shape": list(encoder.projection.weight.shape),
                     "projection_parameters": sum(p.numel() for p in encoder.projection.parameters()),
                     "position_parameters": encoder.position.numel(),
                     "encoder_parameters": sum(p.numel() for p in encoder.parameters())})
    with torch.no_grad():
        assert encoder(image).shape == (1, (16 // patch) ** 2, 16)
encoder4 = VisionEncoder(width=16, patch_size=4)
try:
    encoder4.projection(patchify(image, 2))
except RuntimeError as e:
    result["incompatible_projection"] = {"message": str(e)}
else:
    raise AssertionError("48-column weights must reject a 12-value patch")
assert entrance[1]["encoder_parameters"] - entrance[0]["encoder_parameters"] == 2112
result["entrance"] = entrance

# A hand-designed projection exhibits a possible information difference;
# it is not a trained model or evidence of improved recognition accuracy.
first = torch.zeros(1, 3, 4, 4)
second = torch.zeros_like(first)
first[0, 0, 0, 0] = 1
second[0, 0, 0, 3] = 1
def red_mean_patches(image, patch):
    values = patchify(image, patch)
    return values[..., :patch*patch].mean(-1)
coarse_first, coarse_second = red_mean_patches(first, 4), red_mean_patches(second, 4)
fine_first, fine_second = red_mean_patches(first, 2), red_mean_patches(second, 2)
assert torch.equal(coarse_first, coarse_second)
assert not torch.equal(fine_first, fine_second)
result["possible_local_detail"] = {
    "coarse_first": coarse_first.tolist(), "coarse_second": coarse_second.tolist(),
    "fine_first": fine_first.tolist(), "fine_second": fine_second.tolist(),
    "scope": "Two one-pixel red objects at different locations within the same 4x4 patch become identical under a coarse red-channel-mean projection; 2x2 patch means distinguish their locations. This only establishes a possibility for preserving local detail, not a learned-recognition benefit or lossless patch embedding.",
}

original_path = ROOT / "docs/course-experiments/results/vision_ablation.json"
raw = original_path.read_bytes()
original = json.loads(raw)
pointers = ["/experiment_id", "/revision", "/device", "/seed", "/torch_version", "/python_version", "/step_scale"]
inspection = {p: original[p[1:]] for p in pointers}
inspection["sha256"] = hashlib.sha256(raw).hexdigest()
tok = ByteTokenizer()
variants = {}
assert set(original["results"]["patch_variants"]) == {"4", "8"}
for patch in ["4", "8"]:
    base = f"/results/patch_variants/{patch}"
    variant = original["results"]["patch_variants"][patch]
    train = variant["training"]
    test = variant["test"]
    keys = ["parameters", "trainable_parameters", "config", "modal_config", "steps", "history",
            "effective_tokens", "effective_targets", "weights_changed", "nonzero_gradient_seen", "cpu_smoke"]
    selected_train = {k: train[k] for k in keys}
    pointers.extend(f"{base}/training/{k}" for k in keys)
    samples = test["samples"]
    counters = {}
    for s in samples:
        ids = s["generated_ids"]
        answer_ids = ids[:ids.index(tok.eos_id)] if tok.eos_id in ids else ids
        exact = answer_ids == tok.encode(s["target"])
        assert exact == s["exact_match"]
        assert tok.decode(answer_ids) == s["generated"]
        group = counters.setdefault(s["question"], {"correct": 0, "count": 0})
        group["correct"] += int(exact)
        group["count"] += 1
    correct = sum(s["exact_match"] for s in samples)
    assert len(samples) == test["examples"] == 12
    assert correct == test["correct"] == 9
    assert correct / len(samples) == test["exact_match"] == 0.75
    assert counters == test["groups"]
    assert [h["step"] for h in train["history"]] == list(range(1, 201))
    assert sum(h["effective_targets"] for h in train["history"]) == train["effective_tokens"] == 4765
    assert test["effective_tokens"] == sum(len(tok.encode(s["target"])) + 1 for s in samples) == 72
    keys_test = ["examples", "correct", "exact_match", "effective_tokens", "eos_rate", "generation_errors", "skipped", "samples", "groups"]
    selected_test = {k: test[k] for k in keys_test}
    pointers.extend(f"{base}/test/{k}" for k in keys_test)
    pointers.append(f"{base}/visual_tokens")
    variants[patch] = {"visual_tokens": variant["visual_tokens"], "training": selected_train,
                       "test": selected_test, "recomputed_groups": counters}

data = original["results"]["data"]["splits"]
selected_splits = {}
for split in ["train", "validation", "test"]:
    entry = data[split]
    assert entry["count"] == len(entry["records"])
    rows = entry["records"]
    assert {r["question"] for r in rows} == {"shape?", "color?"}
    assert {r["modality"] for r in rows} == {"vision"}
    assert {r["color"] for r in rows} == {"red", "green", "blue"}
    assert {r["shape"] for r in rows} == {"circle", "square"}
    assert {r["offset"] for r in rows} == {"train": {-2,-1,0}, "validation": {1}, "test": {2}}[split]
    selected_splits[split] = entry
    pointers.append(f"/results/data/splits/{split}")
for left,right in [("train","test"),("train","validation"),("validation","test")]:
    assert not ({r["family"] for r in selected_splits[left]["records"]} & {r["family"] for r in selected_splits[right]["records"]})
for patch in ["4", "8"]:
    for row,sample in zip(selected_splits["test"]["records"], variants[patch]["test"]["samples"], strict=True):
        assert (row["family"],row["question"],row["answer"]) == (sample["family"],sample["question"],sample["target"])
hash_match = {}
for name in ["tiny_perceptron/multimodal.py", "tiny_perceptron/model.py", "tiny_perceptron/attention.py", "scripts/course_experiments/modalities.py"]:
    expected = original["code_sha256"][name]
    observed = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert observed == expected
    hash_match[name] = {"original_sha256": expected, "current_sha256": observed, "matches": True}
    pointers.append("/code_sha256/" + name.replace("~", "~0").replace("/", "~1"))
inspection["checked_pointers"] = pointers
inspection["variants"] = variants
inspection["splits"] = selected_splits
inspection["code_hash_match"] = hash_match
inspection["excluded"] = ["/results/patch_budget", "/results/crop_note", "unrelated intervention results, publication and author-status text"]
result["original_measurement_verification"] = {
    "source_sha256": inspection["sha256"], "variant_keys": ["4", "8"],
    "per_variant": {p: {"correct": v["test"]["correct"], "examples": v["test"]["examples"],
                        "groups": v["recomputed_groups"], "steps": v["training"]["steps"],
                        "train_effective_targets": v["training"]["effective_targets"],
                        "test_effective_targets": v["test"]["effective_tokens"],
                        "parameters": v["training"]["parameters"]} for p,v in variants.items()},
    "split_counts": {s: v["count"] for s,v in selected_splits.items()},
    "inference_scope": "One seed and synthetic shape/color questions only. No patch-2 result or small-text task exists in this inspected comparison. Existing-model measurements were not rerun.",
}
(HERE / "original-measurements-inspected.json").write_text(json.dumps(inspection, ensure_ascii=False, indent=2) + "\n")
(HERE / "cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
