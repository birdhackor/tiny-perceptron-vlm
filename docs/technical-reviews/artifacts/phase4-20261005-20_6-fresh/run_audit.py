"""Inspect specified raw records only; no training or author verdict consumption."""
import ast
import collections
import hashlib
import json
import math
import re
import struct
from pathlib import Path

import torch

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
RAW = ART / "rawdata"
original_dir = ROOT / "outputs/natural-v4/modal-runs/train-37217452291/natural-natural-v4-train-37217452291-1"
original = original_dir / "review/result.json"
report_path = RAW / "training-original-result.json"
assert hashlib.sha256(original.read_bytes()).hexdigest() == hashlib.sha256(report_path.read_bytes()).hexdigest()
report = json.loads(report_path.read_text())
config = json.loads((RAW / "adapter_config.json").read_text())
index = json.loads((RAW / "actual-artifact-index.json").read_text())
indexed = {r["path"]: r for r in index["files"]}
assert hashlib.sha256(report_path.read_bytes()).hexdigest() == indexed["review/result.json"]["sha256"]
assert hashlib.sha256((RAW / "adapter_config.json").read_bytes()).hexdigest() == indexed["review/adapter/adapter_config.json"]["sha256"]

# Only these measurement/provenance pointers are consumed from the original report.
pointers = ["/model", "/model_revision", "/versions", "/dtype", "/lora_rank", "/lora_targets",
            "/learning_rate", "/trainable_parameters", "/completed_steps",
            "/optimizer_parameter_names", "/trainable_parameter_names",
            "/initial_adapter_tensors", "/final_adapter_tensors",
            "/frozen_parameter_samples_initial", "/frozen_parameter_samples_final",
            "/execution/revision", "/execution/runtime_options/learning_rate"]
initial, final = report["initial_adapter_tensors"], report["final_adapter_tensors"]
names = report["optimizer_parameter_names"]
assert len(names) == len(set(names)) == len(initial) == len(final) == 112
assert set(names) == set(initial) == set(final) == set(report["trainable_parameter_names"])
count = sum(math.prod(t["shape"]) for t in final.values())
assert count == report["trainable_parameters"] == 1605632
changed = sum(initial[n]["sha256_values"] != final[n]["sha256_values"] for n in names)
assert changed == 112
assert all(initial[n]["shape"] == final[n]["shape"] and initial[n]["dtype"] == final[n]["dtype"] for n in names)

qwen = json.loads((ART / "sources/qwen-config.json").read_text())["text_config"]
r, layers = config["r"], qwen["num_hidden_layers"]
in_dim = qwen["hidden_size"]
q_out = qwen["num_attention_heads"] * qwen["head_dim"]
v_out = qwen["num_key_value_heads"] * qwen["head_dim"]
expected = layers * r * ((in_dim + q_out) + (in_dim + v_out))
assert (r, config["lora_alpha"], layers, in_dim, q_out, v_out, expected) == (8, 16, 28, 2048, 2048, 1024, 1605632)
assert config["bias"] == "none" and config["lora_bias"] is False and config["use_rslora"] is False
pattern = config["target_modules"]
assert pattern == report["lora_targets"]
expected_names = set()
for layer in range(layers):
    for projection, out_dim in (("q_proj", q_out), ("v_proj", v_out)):
        stem = f"base_model.model.model.language_model.layers.{layer}.self_attn.{projection}"
        assert re.fullmatch(pattern, stem)
        for letter, shape in (("A", [r, in_dim]), ("B", [out_dim, r])):
            name = f"{stem}.lora_{letter}.default.weight"
            expected_names.add(name)
            assert final[name]["shape"] == shape
assert expected_names == set(names)
assert not re.fullmatch(pattern, "model.visual.blocks.0.attn.q_proj")
assert not re.fullmatch(pattern, "model.language_model.layers.0.self_attn.k_proj")

before, after = report["frozen_parameter_samples_initial"], report["frozen_parameter_samples_final"]
assert before == after and len(before) == 7
positions = sum(len(v["flat_indices"]) for v in before.values())
for value in before.values():
    dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32, "torch.float16": torch.float16}[value["dtype"]]
    tensor = torch.tensor(value["values"], dtype=dtype)
    assert list(tensor.shape) == value["shape"]
    assert hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest() == value["sha256_values"]

# Read existing bytes to verify the receipt and final hash measurements; retain the header only.
weight = original_dir / "review/adapter/adapter_model.safetensors"
weight_bytes = weight.read_bytes()
weight_hash = hashlib.sha256(weight_bytes).hexdigest()
weight_record = indexed["review/adapter/adapter_model.safetensors"]
assert len(weight_bytes) == weight_record["bytes"] == 6438952
assert weight_hash == weight_record["sha256"] == "904f4eada21423b06303d8c2a0173b5431e80719557dc7f79ae56ef33a6142cc"
header_length = struct.unpack("<Q", weight_bytes[:8])[0]
header_bytes = weight_bytes[8:8 + header_length]
(RAW / "adapter-safetensors-header.json").write_bytes(header_bytes)
header = json.loads(header_bytes)
tensor_headers = {n: v for n, v in header.items() if n != "__metadata__"}
assert len(tensor_headers) == 112
for name, value in final.items():
    saved = name.replace(".default.", ".")
    t = tensor_headers[saved]
    assert t["shape"] == value["shape"] and t["dtype"] == "F32" and value["dtype"] == "torch.float32"
    begin, end = t["data_offsets"]
    assert end - begin == math.prod(t["shape"]) * 4
    block = weight_bytes[8 + header_length + begin:8 + header_length + end]
    assert hashlib.sha256(block).hexdigest() == value["sha256_values"]
assert 8 + header_length + count * 4 == len(weight_bytes)

# Compare original runtime with today's implementation at each inspected contract.
runtime = ART / "inputs/runtime-natural-assistant-a008cec6.py"
current = ROOT / "tiny_perceptron/natural_assistant.py"
def method(p, name):
    return next(n for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
methods = ("load_core", "run_train", "parameter_counts", "tensor_values", "adapter_tensor_hashes", "frozen_parameter_samples", "load_asr")
for name in methods:
    assert ast.dump(method(runtime, name), include_attributes=False) == ast.dump(method(current, name), include_attributes=False)
assert report["execution"]["revision"] == "a008cec6fd447ca977503dbfd1130cd59a7c7a49"
assert report["learning_rate"] == report["execution"]["runtime_options"]["learning_rate"] == 3e-5

results = {
    "read_pointers": pointers,
    "training_report_original_path": str(original.relative_to(ROOT)),
    "training_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "training_report_copy_same_sha256": True,
    "original_run_revision": report["execution"]["revision"],
    "original_training_versions": report["versions"],
    "inspected_current_runtime_methods_AST_identical": list(methods),
    "language_layers": layers, "rank": r, "alpha": config["lora_alpha"],
    "input_dimension": in_dim, "q_output_dimension": q_out, "v_output_dimension": v_out,
    "parameter_count_formula": "28*8*((2048+2048)+(2048+1024))",
    "recomputed_trainable_parameters": count,
    "optimizer_tensor_count": len(names), "recomputed_changed_value_hashes": changed,
    "all_final_value_hashes_match_existing_safetensors": True,
    "frozen_sample_tensor_count": len(before), "frozen_sample_index_count": positions,
    "frozen_samples_identical_and_internal_hashes_valid": True,
    "learning_rate": report["learning_rate"],
    "adapter_file_bytes": len(weight_bytes), "adapter_file_decimal_MB": len(weight_bytes) / 1e6,
    "adapter_tensor_payload_bytes": count * 4, "header_and_length_bytes": 8 + header_length,
    "adapter_file_sha256": weight_hash,
    "retained_weights": False, "retained_header_sha256": hashlib.sha256(header_bytes).hexdigest(),
    "scope": "Recomputed existing audit measurements; no model reload, training rerun, whole-base comparison, or new capability score. Plain fp32 adapter storage is independent of frozen base dtype/quantization. No percentage denominator is inferred.",
}
(ART / "audit-results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
