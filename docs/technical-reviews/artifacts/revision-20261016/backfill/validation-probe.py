"""Independent CPU inspection of saved scores; never runs inference or training."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).with_name("validation-probe-output.json")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


def predicted(scores):
    return max(range(len(scores)), key=lambda i: scores[i])


def meaning(row):
    return min(zip(row["candidates"], row["raw_scores"]), key=lambda x: (-x[1], x[0]))[0]


def ce(scores, gold, temperature=1):
    values = [v / temperature for v in scores]
    high = max(values)
    return high + math.log(sum(math.exp(v - high) for v in values)) - values[gold]


def run():
    report = load("docs/lesson-products/results/final-report-v1.json")
    manifest = load("docs/lesson-products/data-v2-manifest.json")
    freeze = load("docs/lesson-products/results/settings-freeze-v1.json")
    result = {
        "reviewer_task": "/root/backfill_tech_validation",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "command": "uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/backfill/validation-probe.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "scope": "Saved raw scores and artifacts only. No model inference, parameter updates, GPU jobs, formal evaluation restart, Modal, or HF writes.",
        "inputs": {},
    }
    paths = ["docs/lesson-products/results/final-report-v1.json", "docs/lesson-products/data-v2-manifest.json", "docs/lesson-products/results/settings-freeze-v1.json", "outputs/lesson-products/final-report-v1.predictions.jsonl", "outputs/lesson-products/data-v2/b.jsonl", "outputs/lesson-products/data-v2/c.jsonl"]
    for p in paths:
        result["inputs"][p] = {"sha256": sha(ROOT / p), "bytes": (ROOT / p).stat().st_size}
    assert result["inputs"][paths[3]]["sha256"] == report["predictions"]["sha256"]
    for name in ("b", "c"):
        assert result["inputs"][f"outputs/lesson-products/data-v2/{name}.jsonl"]["sha256"] == manifest["files"][f"{name}.jsonl"]["sha256"]
    b_rows = {}
    family_splits = defaultdict(set)
    observations = {}
    duplicate_example = defaultdict(set)
    for product in ("b", "c"):
        rows = 0
        with (ROOT / f"outputs/lesson-products/data-v2/{product}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                rows += 1
                if row["split"] not in ("train", "calibration", "policy", "test"):
                    continue
                family_splits[row["family_id"]].add(row["split"])
                if product == "b" and row["split"] == "test":
                    b_rows[row["id"]] = row
                if product == "c":
                    key = (row["state"], row["question"], tuple(sorted(row["candidates"])))
                    item = observations.setdefault(key, [set(), set(), 0])
                    item[0].add(row["split"])
                    item[1].add(row["gold"])
                    item[2] += 1
                    if row["state"] == "卡片進度;申請編號=?":
                        duplicate_example[row["question"]].add((row["split"], row["gold"]))
        assert rows == manifest["files"][f"{product}.jsonl"]["rows"]
    duplicates = [v for v in observations.values() if len(v[0]) > 1]
    audit = {"cross_split_shared_observations": len(duplicates), "rows_in_shared_observations": sum(v[2] for v in duplicates), "conflicting_labels": sum(len(v[1]) > 1 for v in observations.values()), "joint_family_overlap": sum(len(v) > 1 for v in family_splits.values()), "example": {k: sorted(v) for k, v in duplicate_example.items()}}
    assert audit["cross_split_shared_observations"] == 5453
    assert audit["rows_in_shared_observations"] == 215427
    assert audit["conflicting_labels"] == audit["joint_family_overlap"] == 0
    result["data_audit"] = audit
    counters = defaultdict(Counter)
    c_ce = defaultdict(list)
    with (ROOT / paths[3]).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["product"] == "b":
                split = row["split"]
                if split == "test":
                    original = b_rows[row["id"]]
                    assert row["gold"] == original["gold"]
                    for depth, scores in row["scores_by_depth"].items():
                        pred = predicted(scores)
                        assert pred == row["answers_by_depth"][depth]
                        for group in [row["task"]] + (["repeated_swap"] if row["task"] == "boxes" and original["strata"]["repeated_swap"] else []):
                            counters[f"b/{depth}/{group}"]["count"] += 1
                            counters[f"b/{depth}/{group}"]["correct"] += pred == row["gold"]
                else:
                    counters[f"b/{split}/{row['task']}"]["count"] += 1
                    counters[f"b/{split}/{row['task']}"]["correct"] += predicted(row["raw_scores"]) == row["gold"]
            elif row["product"] == "c":
                pred = meaning(row)
                key = f"c/{row['split']}/{row['question']}"
                counters[key]["count"] += 1
                counters[key]["correct"] += pred == row["gold"]
                if row["split"] == "test":
                    assert pred == row["selected"]
                    if row["question"] == "choice":
                        counters["c/reorder"]["count"] += 1
                        counters["c/reorder"]["consistent"] += row["reverse_selected"] == pred
                if row["split"] == "wording_shift" and row["question"] == "choice":
                    g = row["candidates"].index(row["gold"])
                    c_ce["before"].append(ce(row["raw_scores"], g))
                    c_ce["after"].append(ce(row["raw_scores"], g, freeze["calibration"]["temperatures"]["choice"]))
    for key, correct, count in [("b/5/boxes", 2928, 2956), ("b/5/sum", 2, 42), ("b/5/repeated_swap", 32, 42), ("b/entity_shift/boxes", 1486, 2956), ("c/test/choice", 11797, 11797), ("c/wording_shift/choice", 11493, 11797)]:
        assert counters[key]["correct"] == correct and counters[key]["count"] == count, (key, counters[key])
    assert dict(counters["c/reorder"]) == {"count": 11797, "consistent": 11797}
    result["raw_score_counts"] = dict(counters)
    result["c_wording_choice_ce"] = {k: sum(v) / len(v) for k, v in c_ce.items()}
    for k, v in result["c_wording_choice_ce"].items():
        assert abs(v - report["shifts"]["c_wording"]["choice"][k]["cross_entropy"]) < 2e-6
    assert result["c_wording_choice_ce"]["after"] > result["c_wording_choice_ce"]["before"]
    training = {}
    for name, directory in [("b", "products/b"), ("c", "products/c"), ("fixed3", "mechanisms/fixed3"), ("odd", "mechanisms/odd")]:
        base = ROOT / "outputs/lesson-products" / directory
        receipt = json.loads((base / "training-receipt.json").read_text())
        checkpoint = torch.load(base / "last.pt", weights_only=True, map_location="cpu")
        model_count = sum(v.numel() for v in checkpoint["model"].values())
        assert model_count == receipt["parameters"]
        assert checkpoint["metadata"]["step"] == receipt["last_step"]
        assert {"model", "optimizer", "sampler_rng", "torch_rng", "cuda_rng", "config", "metadata"} <= set(checkpoint)
        assert receipt["weights_changed"] and receipt["resume_verification"]["exact_next_step"]
        assert receipt["environment"]["gpu"] == "NVIDIA L4"
        training[name] = {k: receipt[k] for k in ("parameters", "last_step", "steps_this_attempt", "selected_step", "initial_weights_sha256", "weights_changed", "environment", "resume_verification")}
        training[name]["receipt_sha256"] = sha(base / "training-receipt.json")
        training[name]["checkpoint_sha256"] = sha(base / "last.pt")
        training[name]["checkpoint_keys"] = list(checkpoint)
    assert training["b"]["last_step"] == 1000 and training["c"]["last_step"] == 800
    assert training["fixed3"]["last_step"] == training["odd"]["last_step"] == 1500
    assert training["fixed3"]["initial_weights_sha256"] == training["odd"]["initial_weights_sha256"]
    result["training_primary_receipts_and_checkpoint_structure"] = training
    result["freeze_git_commit"] = subprocess.check_output(["git", "log", "-1", "--format=%H %cI", "--", "docs/lesson-products/results/settings-freeze-v1.json"], cwd=ROOT, text=True).strip()
    result["formal_test_start"] = load("docs/lesson-products/results/final-report-v1.started.json")
    old = load("docs/selftrained/results/v2-final-public-results.json")
    old_result = {}
    for architecture in ("moe", "dense"):
        p = f"docs/selftrained/results/public-raw/{architecture}/test/metrics.json"
        raw = load(p)
        archived = old["architectures"][architecture]["final_test"]
        assert sha(ROOT / p) == archived["metrics_sha256"]
        assert raw["count"] == archived["count"] == 3734 and raw["evaluation_complete"]
        for k in ("per_task_final_reply", "voice_topic_continuation", "thresholds"):
            assert raw[k] == archived[k]
        old_result[architecture] = {"metrics_sha256": sha(ROOT / p), "count": raw["count"], "tool_roundtrip": raw["per_task_final_reply"]["tool_call"]["tool_roundtrip"], "voice_qa": raw["per_task_final_reply"]["voice_qa"]["semantic"], "ocr": raw["per_task_final_reply"]["ocr"]["exact"], "clothing": raw["per_task_final_reply"]["vision_clothing"]["exact"], "voice_continuation": raw["voice_topic_continuation"], "capability_acceptance_passed": old["architectures"][architecture]["capability_acceptance_passed"]}
    result["old_v2_primary_results"] = old_result
    public_calls = []
    for p in sorted((ROOT / "docs/selftrained/results/public-cpu-raw").glob("*-result.json")):
        item = json.loads(p.read_text())
        assert item["returncode"] == 0
        public_calls.append({"path": str(p.relative_to(ROOT)), "sha256": sha(p), "chat": item["chat"], "returncode": item["returncode"]})
    assert len(public_calls) == 9 and sum(item["chat"] for item in public_calls) == 8
    result["public_cpu_calls"] = public_calls
    result["status"] = "all_assertions_passed"
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(OUT.relative_to(ROOT)), "data_audit": {k:v for k,v in audit.items() if k != "example"}, "c_wording_choice_ce": result["c_wording_choice_ce"]}, ensure_ascii=False))


if __name__ == "__main__":
    run()
