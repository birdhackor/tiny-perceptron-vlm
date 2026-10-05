"""Check immutable recorded measurements only; never load or score a model."""
import hashlib
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "sources"
read_pointers = {}
def load(name):
    return json.loads((SOURCE / name).read_bytes())
cache = load("original-cache.json")
bench = load("original-benchmark.json")
report = load("original-report.json")
data = load("original-data.json")
read_pointers["original-cache.json"] = ["/count", "/selection", "/atol", "/rtol", "/max_new_tokens", "/records/*/id", "/records/*/task", "/records/*/full_trace", "/records/*/cached_trace", "/records/*/generated_ids_equal", "/records/*/same_history_logit_comparisons", "/all_generated_ids_equal", "/all_logits_close", "/diagnostic_stage", "/checkpoint_sha256"]
read_pointers["original-benchmark.json"] = ["/row_id", "/task", "/row", "/selection", "/device", "/dtype", "/max_new_tokens", "/modes/*/warmup_iterations", "/modes/*/measured_iterations", "/modes/*/seconds", "/modes/*/median_seconds", "/modes/*/iterations", "/all_generated_ids_equal", "/generated_ids_equal_by_iteration", "/optimizer_updates", "/weights_unchanged", "/timing_scope", "/checkpoint_sha256", "/dataset_manifest_sha256", "/diagnostic_stage"]
read_pointers["original-report.json"] = ["/revision", "/device", "/gpu", "/torch_version", "/python_version", "/artifacts", "/results/recommended_stage", "/code_sha256/scripts/course_experiments/capstone_deployment.py", "/code_sha256/tiny_perceptron/capstone.py"]
read_pointers["original-data.json"] = ["/schema_version", "/manifest", "/splits/validation"]
first = {}
for row in data["splits"]["validation"]:
    first.setdefault(row["task"], row)
assert len(first) == cache["count"] == len(cache["records"]) == 12
assert [row["id"] for row in cache["records"]] == [first[task]["id"] for task in sorted(first)]
assert cache["max_new_tokens"] == bench["max_new_tokens"] == 16
assert cache["atol"] == cache["rtol"] == 1e-4
assert cache["diagnostic_stage"] == bench["diagnostic_stage"] == report["results"]["recommended_stage"] == "joint"
assert cache["checkpoint_sha256"] == bench["checkpoint_sha256"]
assert bench["row"] == first["style"]
manifest_digest = hashlib.sha256(json.dumps(data["manifest"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
assert manifest_digest == bench["dataset_manifest_sha256"]
for entry in report["artifacts"]:
    lookup = {"cache-consistency.json": "original-cache.json", "generation-benchmark.json": "original-benchmark.json", "data.json": "original-data.json"}
    if entry["path"] in lookup:
        raw = (SOURCE / lookup[entry["path"]]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert len(raw) == entry["bytes"]
for path, file in [("scripts/course_experiments/capstone_deployment.py", "original-deployment.py"), ("tiny_perceptron/capstone.py", "original-capstone.py")]:
    assert hashlib.sha256((SOURCE / file).read_bytes()).hexdigest() == report["code_sha256"][path]
def decode(ids):
    return bytes(i - 8 for i in ids if i >= 8).decode("utf-8", errors="replace")
rows = []
for record in cache["records"]:
    full, cached = record["full_trace"], record["cached_trace"]
    assert full["generated_ids"] == cached["generated_ids"]
    assert full["prompt_ids"] == cached["prompt_ids"]
    assert full["raw"] == decode(full["generated_ids"])
    assert cached["raw"] == decode(cached["generated_ids"])
    assert full["stop_reason"] == cached["stop_reason"]
    assert 0 < len(full["generated_ids"]) <= 16
    comparisons = record["same_history_logit_comparisons"]
    assert len(comparisons) == len(full["generated_ids"])
    for step, comparison in enumerate(comparisons):
        assert comparison["step"] == step
        assert comparison["full_next_id"] == comparison["cached_next_id"] == full["generated_ids"][step]
        assert comparison["allclose"]
        # Since max absolute error is below atol, the recorded raw bound alone
        # suffices for every candidate even without storing all 264 logits.
        assert comparison["max_abs_logit_difference"] <= cache["atol"]
    row = first[record["task"]]
    rows.append({"id": record["id"], "task": record["task"], "image": row["image"] is not None, "audio": row["audio"] is not None,
                 "generated_ids": full["generated_ids"], "stop_reason": full["stop_reason"], "comparisons": len(comparisons),
                 "max_abs_difference": max(c["max_abs_logit_difference"] for c in comparisons)})
assert cache["all_generated_ids_equal"] and cache["all_logits_close"]
modes = {}
expected_ids = [b + 8 for b in b"DIRECT:15"] + [2]
for name in ["full", "cache"]:
    mode = bench["modes"][name]
    assert mode["warmup_iterations"] == 3 and mode["measured_iterations"] == 10
    assert len(mode["iterations"]) == 13
    measured = []
    for i, record in enumerate(mode["iterations"]):
        assert record["iteration"] == i
        assert record["phase"] == ("warmup" if i < 3 else "measured")
        assert record["trace"]["generated_ids"] == expected_ids
        assert record["trace"]["raw"] == decode(expected_ids) == "DIRECT:15"
        assert record["stop_reason"] == record["trace"]["stop_reason"] == "eos"
        assert record["eos"] and record["trace"]["eos"]
        assert record["generated_token_count_including_eos"] == 10
        if i >= 3:
            measured.append(record["seconds"])
    assert measured == mode["seconds"]
    median = statistics.median(measured)
    assert median == mode["median_seconds"]
    modes[name] = {"measured_count": len(measured), "warmup_count": 3, "median_seconds": median, "median_ms": median * 1000,
                   "rounded_median_ms": round(median * 1000, 3), "raw_seconds": measured}
assert modes["full"]["rounded_median_ms"] == 68.054
assert modes["cache"]["rounded_median_ms"] == 59.978
assert bench["dtype"] == "float32" and bench["device"] == "cuda:0"
assert report["gpu"] == "NVIDIA L4"
assert bench["all_generated_ids_equal"] and len(bench["generated_ids_equal_by_iteration"]) == 13 and all(bench["generated_ids_equal_by_iteration"])
assert bench["optimizer_updates"] == 0 and bench["weights_unchanged"]
result = {"scope": "Independent arithmetic and provenance check of existing immutable GPU measurements; no new inference or training.",
          "source_commit": "1df335318bda03fd771807f66976953231d5a00b", "recorded_runtime_revision": report["revision"],
          "recorded_environment": {k: report[k] for k in ["device", "gpu", "torch_version", "python_version"]},
          "checkpoint_sha256": cache["checkpoint_sha256"], "validation_rows": len(data["splits"]["validation"]),
          "cache_records": rows, "compared_positions": sum(x["comparisons"] for x in rows),
          "cache_atol": cache["atol"], "cache_rtol": cache["rtol"], "generation_modes": modes,
          "benchmark_ids": expected_ids, "benchmark_prompt": bench["row"]["user"], "timing_scope": bench["timing_scope"],
          "read_json_pointers": read_pointers, "passed": True}
(BASE / "recorded-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
