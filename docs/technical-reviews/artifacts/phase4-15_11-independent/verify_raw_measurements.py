"""Read named original measurement/provenance pointers only, never author verdicts."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = (HERE / "primary/moe-original.json").read_bytes()
original = json.loads(raw)
selected, pointers = {}, []


def read(pointer):
    value = original
    for key in pointer.strip("/").split("/"):
        value = value[int(key)] if isinstance(value, list) else value[key]
    pointers.append(pointer)
    selected[pointer] = value
    return value


for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale"):
    read("/" + key)
for key in ("tiny_perceptron/modern.py", "scripts/course_experiments/architecture.py", "scripts/course_experiments/common.py"):
    # The code_sha256 keys contain slashes, so access this specific dictionary member directly.
    pointer = "/code_sha256/" + key.replace("/", "~1")
    pointers.append(pointer)
    selected[pointer] = original["code_sha256"][key]
dataset = read("/results/dataset")
assert [dataset[k]["records"] for k in ("train", "validation", "test")] == [409, 51, 52]
names = ("dense_active_top1", "dense_active_top2", "dense_total", "top1_aux0.01", "top2_aux0.01")
expected = ((64,12.676,35.693,112.648),(80,11.965,67.945,119.091),(104,14.054,68.413,129.432),
            (64,27.630,68.454,120.711),(64,27.242,68.454,133.961))
fields = ("requested_steps", "steps", "optimizer_updates", "skipped_updates", "batch_size", "effective_tokens",
          "auxiliary_weight", "gradient_clip_norm", "warm_step_median_seconds", "memory_allocated_before_bytes",
          "peak_memory_allocated_bytes", "peak_additional_allocated_bytes", "scaler_enabled", "all_parameters_finite")
rows = []
for name, (width, milliseconds, before_mib, peak_mib) in zip(names, expected):
    prefix = "/results/variants/" + name
    config = read(prefix + "/model/config")
    tr = {key: read(prefix + "/training/" + key) for key in fields}
    assert config["width"] == width and config["max_length"] == 128
    assert tr["requested_steps"] == tr["steps"] == tr["optimizer_updates"] == 180
    assert tr["skipped_updates"] == 0 and tr["batch_size"] == 16 and tr["effective_tokens"] == 337761
    assert tr["peak_additional_allocated_bytes"] == tr["peak_memory_allocated_bytes"] - tr["memory_allocated_before_bytes"]
    conversions = (tr["warm_step_median_seconds"] * 1000, tr["memory_allocated_before_bytes"] / 2**20,
                   tr["peak_memory_allocated_bytes"] / 2**20)
    assert [round(x, 3) for x in conversions] == [milliseconds, before_mib, peak_mib]
    assert all(abs(x-y) <= 0.0005 for x,y in zip(conversions, (milliseconds,before_mib,peak_mib)))
    assert tr["auxiliary_weight"] == (0.01 if name.startswith("top") else 0)
    assert tr["all_parameters_finite"] is True
    rows.append({"variant": name, "milliseconds": conversions[0], "before_MiB": conversions[1],
                 "peak_MiB": conversions[2], "rounded_table": [round(x, 3) for x in conversions]})
assert original["gpu"] == "NVIDIA L4"
recorded_code = (HERE / "primary/architecture-recorded-revision.py").read_bytes()
assert hashlib.sha256(recorded_code).hexdigest() == original["code_sha256"]["scripts/course_experiments/architecture.py"]
result = {"raw_original_sha256": hashlib.sha256(raw).hexdigest(), "inspected_pointers": pointers,
          "selected_raw_measurements_and_provenance": selected, "converted_table": rows,
          "denominators": {"variants_in_table":5,"requested_updates_each":180,"successful_updates_each":180,
                           "median_calls_after_first_three_each":177,"batch_sequences":16,"max_sequence_positions":128,
                           "effective_training_tokens_each":337761,"dataset_records":{"train":409,"validation":51,"test":52}},
          "underlying_latency_limit": "Original training JSON stores warm_step_median_seconds, not the 180-element latencies array. The median cannot be independently recalculated; only its original producing formula and conversions were verified. No GPU rerun or missing measurements created.",
          "scope": "Existing observed timing and allocator measurement records only; no heldout scores or author interpretation fields read."}
(HERE / "raw-measurement-verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"converted_table":rows,"denominators":result["denominators"],"recorded_code_sha_matches":True,
                  "median_raw_series_available":False},ensure_ascii=False))
