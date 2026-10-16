"""19.20 independent bounded CPU evidence check; no model or training invocation."""
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).with_name("19.20-probe-output.json")
COMMIT = "821a2c717f05ad7b57e08e88cd965e92ff0c1200"


def read(path):
    return json.loads((ROOT / path).read_bytes())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def choice(row):
    return min(zip(row["candidates"], row["raw_scores"]), key=lambda x: (-x[1], x[0]))[0]


def rate(correct, count):
    return {"correct": correct, "count": count, "percent": 100 * correct / count}


def world_answer(row):
    w = row["world"]
    if row["task"] != "boxes":
        return row["gold"]  # Arithmetic gold linkage checked against original dataset, not redefined.
    a, b = w["a"], w["b"]
    for event in w["events"]:
        if event[0] == "swap":
            a, b = b, a
        elif event[0] == "A":
            a = event[1]
        elif event[0] == "B":
            b = event[1]
        else:
            raise ValueError(event)
    return a if w["query"] == "A" else b


report = read("docs/lesson-products/results/final-report-v1.json")
freeze_path = "docs/lesson-products/results/settings-freeze-v1.json"
freeze = read(freeze_path)
manifest_path = "outputs/lesson-products/data-v2/manifest.json"
manifest = read(manifest_path)
raw_path = "outputs/lesson-products/final-report-v1.predictions.jsonl"
assert sha(raw_path) == report["predictions"]["sha256"]
assert sha(manifest_path) == freeze["data_manifest_sha256"] == report["data_manifest_sha256"]
assert sha(freeze_path) == report["settings_freeze_sha256"]
commit_bytes = subprocess.check_output(["git", "show", COMMIT + ":" + freeze_path], cwd=ROOT)
assert hashlib.sha256(commit_bytes).hexdigest() == sha(freeze_path)
commit_time = subprocess.check_output(["git", "show", "-s", "--format=%cI", COMMIT], cwd=ROOT, text=True).strip()
started = read("docs/lesson-products/results/final-report-v1.started.json")
assert datetime.fromisoformat(commit_time) < datetime.fromisoformat(started["recorded_at"])
file_checks = {}
for product in ("b", "c"):
    for name, expected in freeze["products"][product].items():
        path = "outputs/lesson-products/products/" + product + "/" + name
        actual = sha(path)
        assert actual == expected, path
        file_checks[path] = actual
for mode, directory in (("fixed3", "fixed3"), ("odd", "odd")):
    for name in ("last.pt", "training-receipt.json"):
        path = "outputs/lesson-products/mechanisms/" + directory + "/" + name
        assert sha(path) == freeze["mechanisms"][mode][name]
        file_checks[path] = sha(path)
for name, expected in freeze["source_sha256s"].items():
    path = "tiny_perceptron/lesson_products/" + name
    assert sha(path) == expected
    file_checks[path] = expected
for name, expected in manifest["files"].items():
    path = "outputs/lesson-products/data-v2/" + name
    assert sha(path) == expected["sha256"]
    file_checks[path] = sha(path)

b_data = {}
with (ROOT / "outputs/lesson-products/data-v2/b.jsonl").open() as stream:
    for line in stream:
        row = json.loads(line)
        if row["split"] == "test":
            assert row["gold"] == world_answer(row)
            b_data[row["id"]] = row
c_data = {}
duplicate = defaultdict(dict)
c_families = defaultdict(set)
with (ROOT / "outputs/lesson-products/data-v2/c.jsonl").open() as stream:
    for line in stream:
        row = json.loads(line)
        if row["split"] in ("train", "calibration", "policy", "test"):
            c_families[row["split"]].add(row["family_id"])
            if row["state"] == "卡片進度;申請編號=?":
                duplicate[row["question"]][row["split"]] = row["gold"]
        if row["split"] == "test":
            c_data[row["id"]] = row
assert all(not a.intersection(b) for i, a in enumerate(c_families.values()) for b in list(c_families.values())[i + 1:])
b_counts = Counter()
c_counts = Counter()
repeated = Counter()
forward_vs_stored = Counter()
raw_counts = Counter()
shift_counts = Counter()
with (ROOT / raw_path).open() as stream:
    for line in stream:
        row = json.loads(line)
        raw_counts[(row["product"], row["split"])] += 1
        if row["product"] == "b" and row["split"] == "test":
            data = b_data[row["id"]]
            assert data["gold"] == row["gold"] and data["task"] == row["task"]
            for depth, scores in row["scores_by_depth"].items():
                pred = argmax(scores)
                assert pred == row["answers_by_depth"][depth]
                b_counts[(depth, row["task"], "count")] += 1
                b_counts[(depth, row["task"], "correct")] += pred == row["gold"]
                if data["task"] == "boxes" and data["strata"]["repeated_swap"]:
                    repeated[(depth, "count")] += 1
                    repeated[(depth, "correct")] += pred == row["gold"]
        if row["product"] == "c" and row["split"] == "test":
            data = c_data[row["id"]]
            assert data["gold"] == row["gold"] and data["candidates"] == row["candidates"]
            pred = choice(row)
            assert pred == row["selected"]
            c_counts[(row["question"], "count")] += 1
            c_counts[(row["question"], "correct")] += pred == row["gold"]
            if row["question"] == "choice":
                forward_vs_stored["count"] += 1
                forward_vs_stored["consistent"] += pred == row["reverse_selected"]
        if row["product"] == "c" and row["split"] == "wording_shift" and row["question"] == "choice":
            shift_counts["count"] += 1
            shift_counts["correct"] += choice(row) == row["gold"]

b_curves = {str(d): {task: rate(b_counts[(str(d), task, "correct")], b_counts[(str(d), task, "count")]) for task in ("boxes", "sum")} for d in range(1, 7)}
assert b_curves["5"]["boxes"] == rate(2928, 2956)
assert b_curves["5"]["sum"] == rate(2, 42)
assert repeated[("5", "correct")] == 32 and repeated[("5", "count")] == 42
assert c_counts[("choice", "correct")] == c_counts[("choice", "count")] == 11797
assert forward_vs_stored["consistent"] == forward_vs_stored["count"] == 11797
for d, tasks in b_curves.items():
    for task, value in tasks.items():
        assert value["correct"] == report["b"][d][task]["correct"]
        assert value["count"] == report["b"][d][task]["count"]
assert {q: rate(c_counts[(q, "correct")], c_counts[(q, "count")]) for q in ("choice", "info", "rubric")}

old = read("docs/selftrained/results/v2-final-public-results.json")
old_acceptance = {name: value["capability_acceptance_passed"] for name, value in old["architectures"].items()}
assert old_acceptance == {"moe": False, "dense": False}
public = read("docs/lesson-products/results/publication-package-v1.json")
assert public["files"]["report.json"]["sha256"] == sha("docs/lesson-products/results/final-report-v1.json")
assert "raw/final-report-v1.predictions.jsonl.gz" in public["files"]

result = {
    "reviewer_task": "/root/backfill_tech_19_20",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "command": "python docs/technical-reviews/artifacts/revision-20261016/backfill/19.20-probe.py",
    "environment": {"python": platform.python_version(), "device": "CPU; stdlib-only score aggregation; no model execution", "platform": platform.platform()},
    "source_sha256s": file_checks,
    "freeze_sha256": sha(freeze_path),
    "raw_predictions_sha256": sha(raw_path),
    "raw_predictions_bytes": (ROOT / raw_path).stat().st_size,
    "pretest_commit": {"revision": COMMIT, "committed_at": commit_time, "test_started_at": started["recorded_at"], "same_freeze_bytes": True},
    "b_curves": b_curves,
    "b_repeated_swap_at_5": rate(repeated[("5", "correct")], repeated[("5", "count")]),
    "c": {q: rate(c_counts[(q, "correct")], c_counts[(q, "count")]) for q in ("choice", "info", "rubric")},
    "c_reorder": dict(forward_vs_stored),
    "reorder_evidence_limit": "Forward semantic argmax recomputed from logits; reverse selections are primary saved outputs, reversed logits were not saved. No formal test rerun.",
    "c_same_observation_across_splits": dict(duplicate),
    "c_family_sets_pairwise_disjoint": True,
    "c_wording_shift_choice": rate(shift_counts["correct"], shift_counts["count"]),
    "A_capability_acceptance": old_acceptance,
    "raw_record_counts": {"/".join(k): v for k, v in raw_counts.items()},
    "public_package_files": {k: public["files"][k] for k in ("report.json", "raw/final-report-v1.predictions.jsonl.gz", "raw/predictions-manifest.json", "settings-freeze.json")},
    "own_judgment_before_other_execution_receipts": "Main four acceptance rows and repeated-swap counts match. Pretest committed settings match current model/data/code/temperature/policy. C repeated generic observations confirmed; 100% is finite fixed-grammar scope. A remains failed. Historical first-test/no-retuning is supported by guards, time receipts and unchanged bytes, not a proof against deliberate unrecorded bypass. No other review conclusion was read.",
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ("b_curves", "b_repeated_swap_at_5", "c", "c_reorder", "c_same_observation_across_splits", "c_wording_shift_choice", "A_capability_acceptance")}, ensure_ascii=False, indent=2))
