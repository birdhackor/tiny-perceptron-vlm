"""Recompute fixed GPU-record assertions; this does not run GPU training."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.natural_assistant import training_row_at

torch.set_num_threads(1)
out = Path(__file__).resolve().parent
research = ROOT / "outputs/natural-v4/factual-research/20.6"
run = ROOT / "outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1"
review = run / "review"
t = json.loads((review / "training.json").read_text())
config = json.loads((review / "adapter/adapter_config.json").read_text())
execution = json.loads((review / "execution.json").read_text())
qwen = json.loads((research / "qwen-config.json").read_text())
manifest_path = ROOT / "docs/natural-assistant/v4/manifest.json"
manifest = json.loads(manifest_path.read_text())
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == t["manifest_sha256"]
rows = [row for row in manifest["rows"] if row["split"] == "train"]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Verify all genuinely present original review files against the public size/SHA index.
index_path = ROOT / "docs/natural-assistant/evidence/v4-runtime/train-37217452291/actual-artifact-index.json"
index = json.loads(index_path.read_text())
receipts = []
for record in index["files"]:
    path = run / record["path"]
    assert path.stat().st_size == record["bytes"]
    assert sha(path) == record["sha256"]
    receipts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})

# Original config and pinned Transformers source define all unique parameters.
# Embedding and lm_head weights are tied; count that storage only once.
tc, vc = qwen["text_config"], qwen["vision_config"]
h, kv, q, head, ff, layers = [tc[k] for k in
    ("hidden_size", "num_key_value_heads", "num_attention_heads", "head_dim", "intermediate_size", "num_hidden_layers")]
assert not tc["attention_bias"] and qwen["tie_word_embeddings"]
text_components = {
    "shared_embedding_and_lm_head": tc["vocab_size"] * h,
    "attention_weights_per_layer": h * (q * head + 2 * kv * head) + q * head * h,
    "MLP_weights_per_layer": 3 * h * ff,
    "RMSNorm_weights_per_layer": 2 * h + 2 * head,
    "final_RMSNorm_weights": h,
}
language = text_components["shared_embedding_and_lm_head"] + layers * sum(
    text_components[k] for k in ("attention_weights_per_layer", "MLP_weights_per_layer", "RMSNorm_weights_per_layer")
) + h
v, vf, merge = vc["hidden_size"], vc["intermediate_size"], vc["hidden_size"] * vc["spatial_merge_size"] ** 2
vision_block = 4 * v + (v * 3 * v + 3 * v) + (v * v + v) + (v * vf + vf) + (vf * v + v)
merger_linears = merge * merge + merge + merge * vc["out_hidden_size"] + vc["out_hidden_size"]
vision_components = {
    "patch_embed_weight_and_bias": v * vc["in_channels"] * vc["temporal_patch_size"] * vc["patch_size"] ** 2 + v,
    "position_embedding": vc["num_position_embeddings"] * v,
    "blocks": vc["depth"] * vision_block,
    "main_merger": merger_linears + 2 * v,
    "deepstack_mergers": len(vc["deepstack_visual_indexes"]) * (merger_linears + 2 * merge),
}
vision = sum(vision_components.values())
base = language + vision
rank = config["r"]
adapter_formula = layers * (rank * (h + q * head) + rank * (h + kv * head))
assert base == 2127532032
assert adapter_formula == 1605632
assert base + adapter_formula == t["total_parameters"] == 2129137664
assert adapter_formula == t["trainable_parameters"]
percent = 100 * adapter_formula / (base + adapter_formula)
assert round(percent, 4) == 0.0754
assert rank == 8 and config["lora_alpha"] == 16 and not config["use_rslora"]
assert config["target_modules"] == t["lora_targets"]
assert t["learning_rate"] == execution["runtime_options"]["learning_rate"] == 0.00003

# Parse existing safetensors bytes directly, without loading a large model or installing a package.
weights_path = review / "adapter/adapter_model.safetensors"
raw = weights_path.read_bytes()
header_bytes = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8:8 + header_bytes])
tensors = {name: metadata for name, metadata in header.items() if name != "__metadata__"}
initial, final = t["initial_adapter_tensors"], t["final_adapter_tensors"]
assert set(initial) == set(final) == set(t["optimizer_parameter_names"]) == set(t["trainable_parameter_names"])
assert len(initial) == len(tensors) == 112
changed = 0
tensor_receipts = []
shapes = Counter()
for name, old in initial.items():
    assert re.fullmatch(r"base_model\.model\.model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)\.lora_[AB]\.default\.weight", name)
    new = final[name]
    saved_name = name.replace(".default", "")
    metadata = tensors[saved_name]
    assert old["shape"] == new["shape"] == metadata["shape"]
    assert metadata["dtype"] == "F32" and new["dtype"] == old["dtype"] == "torch.float32"
    begin, end = metadata["data_offsets"]
    data = raw[8 + header_bytes + begin:8 + header_bytes + end]
    assert end - begin == 4 * math.prod(metadata["shape"])
    final_sha = hashlib.sha256(data).hexdigest()
    assert final_sha == new["sha256_values"]
    changed += old["sha256_values"] != final_sha
    shapes[str(metadata["shape"])] += 1
    if ".lora_B." in name:
        assert hashlib.sha256(bytes(len(data))).hexdigest() == old["sha256_values"]
    tensor_receipts.append({"name": name, "shape": metadata["shape"], "elements": math.prod(metadata["shape"]), "initial_sha256": old["sha256_values"], "actual_final_sha256": final_sha})
assert changed == t["changed_adapter_tensor_count"] == 112
elements = sum(math.prod(metadata["shape"]) for metadata in tensors.values())
assert elements == adapter_formula
assert all("lora_" in name for name in tensors)
assert len(raw) == 6438952 == 8 + header_bytes + 4 * elements

frozen_initial, frozen_final = t["frozen_parameter_samples_initial"], t["frozen_parameter_samples_final"]
assert frozen_initial == frozen_final and t["frozen_parameter_samples_unchanged"]
for name, sample in frozen_initial.items():
    assert sample["shape"] == [3] and sample["dtype"] == "torch.bfloat16"
    assert len(sample["flat_indices"]) == len(set(sample["flat_indices"])) == 3
    actual = torch.tensor(sample["values"], dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
    assert hashlib.sha256(actual).hexdigest() == sample["sha256_values"]
assert len(frozen_initial) == 7

history = t["history"]
assert [r["step"] for r in history] == list(range(1, 2078))
seen_ids = []
for update in history:
    expected_ids = [training_row_at(rows, (update["step"] - 1) * 2 + micro, t["seed"])["id"] for micro in range(2)]
    assert update["row_ids"] == expected_ids
    seen_ids.extend(update["row_ids"])
assert len(seen_ids) == t["trained_rows"] == 4154
tokens = sum(r["supervised_tokens"] for r in history)
assert tokens == 78872 and t["seed"] == 42 and t["gradient_accumulation"] == 2
summary = {
    "execution_scope": "Own bounded CPU arithmetic, hash/header comparison and data-record validation of a fixed completed GPU run; no own GPU training or quality replication.",
    "run_id": execution["run_id"], "recorded_GPU_environment": {k:t[k] for k in ("versions", "device", "dtype", "gpu_name")},
    "text_components": text_components, "language_total": language,
    "vision_components": vision_components, "vision_total": vision,
    "base_parameters": base, "adapter_parameters": elements, "total_including_adapter": base + elements,
    "adapter_formula": "28 * (8*(2048+2048) + 8*(2048+1024))",
    "percentage_with_adapter_denominator": percent, "rounded_percentage": round(percent, 4),
    "adapter_tensor_count": len(tensors), "shape_frequencies": dict(shapes),
    "changed_adapter_tensors": changed, "all_final_adapter_tensor_hashes_recomputed_from_existing_bytes": True,
    "adapter_file_bytes": len(raw), "decimal_MB": len(raw)/1_000_000,
    "binary_MiB": len(raw)/1048576, "safetensors_header_bytes": header_bytes,
    "adapter_float32_data_bytes": 4 * elements, "file_sha256": sha(weights_path),
    "adapter_file_has_only_LoRA_tensors_no_base_or_optimizer_state": True,
    "frozen_sample_tensors": len(frozen_initial), "frozen_sample_values": sum(len(s["flat_indices"]) for s in frozen_initial.values()),
    "frozen_sample_records_equal": frozen_initial == frozen_final,
    "frozen_sample_hashes_recomputed_from_disclosed_values": True,
    "frozen_scope": t["frozen_check_scope"], "frozen_samples": frozen_initial,
    "learning_rate": t["learning_rate"], "rank": rank, "alpha": config["lora_alpha"], "alpha_over_rank": config["lora_alpha"]/rank,
    "manifest_sha256": t["manifest_sha256"], "seed": t["seed"], "train_split_rows": len(rows),
    "updates": len(history), "rows_used": len(seen_ids), "unique_rows_used": len(set(seen_ids)),
    "shifted_supervised_tokens_recorded": tokens, "all_update_row_ids_recomputed_from_manifest_order": True,
    "recorded_status": t["status"], "original_review_file_receipts": receipts,
    "tensor_receipts": tensor_receipts,
}
(out / "record-results.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k:v for k,v in summary.items() if k not in {"original_review_file_receipts", "tensor_receipts", "frozen_samples"}},indent=2))
print("FIXED_GPU_RECORD_CHECKS_PASSED")
