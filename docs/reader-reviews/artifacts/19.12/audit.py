"""Read linked, stored evidence on CPU; do not train or generate model answers."""

import json
import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
loaded = {}


def read(relative):
    raw = (ROOT / relative).read_bytes()
    loaded[relative] = sha256(raw).hexdigest()
    return json.loads(raw)


source = (ROOT / "course/chapters/19.md").read_bytes()
section = source[source.index(b"## 19.12 "):]
(OUT / "source-read.md").write_bytes(section)
shortcode = re.search(rb"```python\n(.*?)```", section, re.S).group(1)
(OUT / "shortcode.py").write_bytes(shortcode)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
run = subprocess.run(
    [str(ROOT / ".venv/bin/python"), str(OUT / "shortcode.py")],
    cwd=ROOT, env=environment, capture_output=True, text=True, check=True,
)
(OUT / "shortcode.stdout.txt").write_text(run.stdout)

data = read("docs/course-experiments/capstone-evidence/deployment/data.json")
test = {row["id"]: row for row in data["splits"]["test"]}
counts = dict(Counter(row["task"] for row in test.values()))
assert len(test) == 90
deployment = read("docs/course-experiments/results/capstone_deployment.json")
student = read("docs/course-experiments/results/capstone_student.json")
selection = read("docs/course-experiments/capstone-selection.json")


def check_trace(trace):
    ids = trace["generated_ids"]
    eos_from_ids = bool(ids and ids[-1] == 2)
    assert trace["eos"] == eos_from_ids
    assert (trace["stop_reason"] == "eos") == eos_from_ids
    # Stored IDs use eight reserved codes and 256 byte codes. Independently
    # reconstruct all visible text, including the replacement character.
    decoded = bytes(i - 8 for i in ids if 8 <= i < 264).decode("utf-8", errors="replace")
    assert decoded == trace["raw"]
    return eos_from_ids


def audit_file(relative, original=True):
    evidence = read(relative)
    by_task = defaultdict(lambda: {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
    seen = set()
    for record in evidence["records"]:
        assert record["id"] not in seen
        seen.add(record["id"])
        if original:
            row = test[record["id"]]
            assert record["expected_action"] == row["answer"]
            assert record["task"] == row["task"]
        action = record["action_trace"]
        action_ok = check_trace(action) and action["raw"] == record["expected_action"]
        if record["task"] == "calculator":
            runtime = record["runtime"]
            final = record["final_trace"]
            if runtime and runtime["status"] == "ok":
                match = re.fullmatch(r"TOOL:calculator:(\d+)\+(\d+)", action["raw"])
                assert match is not None
                assert runtime["result"] == str(sum(map(int, match.groups())))
            final_ok = bool(final and check_trace(final) and final["raw"] == "DIRECT:" + record["expected_final"])
            correct = bool(action_ok and runtime and runtime["status"] == "ok" and final_ok)
        else:
            correct = action_ok
            assert record["runtime"] is None
            assert record["final_trace"] is None
        assert action_ok == record["action_correct"]
        assert correct == record["end_to_end_correct"]
        total = by_task[record["task"]]
        total["count"] += 1
        total["action_correct"] += action_ok
        total["end_to_end_correct"] += correct
    if original:
        assert seen == set(test)
    assert dict(by_task) == evidence["by_task"]
    result = {name: sum(task[name] for task in by_task.values()) for name in ["count", "action_correct", "end_to_end_correct"]}
    assert all(result[name] == evidence[name] for name in result)
    return result | {"by_task": dict(by_task)}, evidence


files = {
    "untrained": "deployment/test-untrained.json",
    "pretrain": "deployment/test-pretrain.json",
    "sft": "deployment/test-sft.json",
    "joint": "deployment/test-joint.json",
    "dpo": "deployment/test-dpo.json",
    "joint-int4": "deployment/test-joint-ptq4.json",
    "joint-int8": "deployment/test-joint-ptq8.json",
    "dpo-int4": "deployment/test-ptq4.json",
    "dpo-int8": "deployment/test-ptq8.json",
    "ce": "student/test-ce.json",
    "kd": "student/test-kd.json",
    "kd-int4": "student/test-kd-ptq4.json",
}
results = {}
records = {}
for name, relative in files.items():
    result, evidence = audit_file("docs/course-experiments/capstone-evidence/" + relative)
    results[name] = result
    records[name] = evidence
    summary = deployment["results"]["stages"].get(name)
    if summary is None:
        key = {"ce": "test_ce", "kd": "test_kd", "kd-int4": "test_kd_ptq4"}[name]
        summary = student["results"]["evaluations"][key]
    assert all(summary[key] == result[key] for key in result)

joint = records["joint"]
nonrefusal = [r for r in joint["records"] if r["task"] != "safety"]
assert len(nonrefusal) == 87
assert sum(r["end_to_end_correct"] for r in nonrefusal) == 75
assert all("不能提供他人密碼" not in r["action_trace"]["raw"] for r in nonrefusal)
assert all(row["image"]["color"] == "green" for row in test.values() if row["task"] == "image_color")

image_result, image = audit_file("docs/course-experiments/capstone-evidence/deployment/test-joint-image-swaps.json", original=False)
audio_result, audio = audit_file("docs/course-experiments/capstone-evidence/deployment/test-joint-audio-swaps.json", original=False)
pairs = read("docs/course-experiments/capstone-evidence/deployment/test-joint-image-pairs.json")
original_rows = {r["id"]: r for r in joint["records"]}
swapped_rows = {r["id"]: r for r in image["records"]}
pair_counts = defaultdict(lambda: {"count": 0, "both_correct": 0})
for pair in pairs["pairs"]:
    first = original_rows[pair["original_id"]]
    second = swapped_rows[pair["swapped_id"]]
    both = first["end_to_end_correct"] and second["end_to_end_correct"]
    assert both == pair["both_end_to_end_correct"]
    pair_counts[first["task"]]["count"] += 1
    pair_counts[first["task"]]["both_correct"] += both
assert sum(v["both_correct"] for v in pair_counts.values()) == 27
assert sum(original_rows[r["id"].removesuffix("-audio-swap")]["end_to_end_correct"] and r["end_to_end_correct"] for r in audio["records"]) == 18

cache = read("docs/course-experiments/capstone-evidence/deployment/cache-consistency.json")
assert len(cache["records"]) == 12
assert all(r["full_trace"]["generated_ids"] == r["cached_trace"]["generated_ids"] for r in cache["records"])
bench = read("docs/course-experiments/capstone-evidence/deployment/generation-benchmark.json")
timings = {}
for mode, value in bench["modes"].items():
    measured = [r["seconds"] for r in value["iterations"] if r["phase"] == "measured"]
    assert len(measured) == 10
    assert measured == value["seconds"]
    timings[mode] = statistics.median(measured) * 1000
    assert abs(statistics.median(measured) - value["median_seconds"]) < 1e-12

generation_changes = {}
for first, second in [("joint", "joint-int4"), ("joint", "joint-int8"), ("dpo", "dpo-int4"), ("dpo", "dpo-int8"), ("kd", "kd-int4")]:
    a = {r["id"]: r for r in records[first]["records"]}
    changed = []
    for b in records[second]["records"]:
        initial = a[b["id"]]
        for key in ["action_trace", "final_trace"]:
            ids_a = initial[key]["generated_ids"] if initial[key] else None
            ids_b = b[key]["generated_ids"] if b[key] else None
            if ids_a != ids_b:
                changed.append(b["id"])
                break
    generation_changes[f"{first}->{second}"] = changed

assert selection["selected_stage"] == "joint"
assert selection["selected_before_test_generation"] is True
assert student["results"]["teacher_checkpoint_sha256"] == selection["candidates"]["dpo"]["checkpoint_sha256"]
assert student["results"]["branches"]["ce"]["parameters"]["parameters"] == 79920
assert student["results"]["branches"]["kd"]["parameters"]["parameters"] == 79920

audit = {
    "source_sha256": sha256(section).hexdigest(),
    "shortcode_cpu_stdout": run.stdout,
    "test_counts": counts,
    "recomputed_versions": results,
    "image_pairs_by_task": dict(pair_counts),
    "image_swap": image_result,
    "audio_swap": audio_result,
    "nonrefusal_correct": [75, 87],
    "cache_generated_ids_equal": 12,
    "reported_L4_median_ms_recomputed": timings,
    "generation_changed_ids": generation_changes,
    "evidence_sha256": loaded,
    "scope": "CPU inspection of saved evidence and exact short code; no training, GPU work, new model inference, or human reader study",
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"stage_correct": {k: v["end_to_end_correct"] for k, v in results.items()}, "timings_ms": timings, "generation_changes": {k: len(v) for k, v in generation_changes.items()}, "source_sha256": audit["source_sha256"]}, ensure_ascii=False, indent=2))
