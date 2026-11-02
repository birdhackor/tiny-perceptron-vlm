"""19.20 independent CPU audit of preserved 2026-10-16 records; no model inference/training."""

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import torch

from tiny_perceptron.lesson_products.data import audit_rows, build_rows, canonical


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-gzip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    results = root / "docs/lesson-products/results"
    report = json.loads((results / "final-report-v1.json").read_bytes())
    frozen = json.loads((results / "settings-freeze-v1.json").read_bytes())
    manifest = json.loads((root / "docs/lesson-products/data-v2-manifest.json").read_bytes())
    pin = "821a2c717f05ad7b57e08e88cd965e92ff0c1200"
    pin_time = subprocess.check_output(["git", "show", "-s", "--format=%cI", pin], cwd=root, text=True).strip()
    identities = {}
    for rel in [
        "docs/lesson-products/results/settings-freeze-v1.json",
        "docs/lesson-products/results/pretest-freeze-record-v1.json",
        "docs/lesson-products/results/settings-selection-v1.json",
        "docs/lesson-products/data-v2-manifest.json",
        *["tiny_perceptron/lesson_products/" + name for name in frozen["source_sha256s"]],
    ]:
        historic = subprocess.check_output(["git", "show", pin + ":" + rel], cwd=root)
        identities[rel] = {
            "sha256": sha(root / rel),
            "pretest_sha256": hashlib.sha256(historic).hexdigest(),
            "same_as_pretest": historic == (root / rel).read_bytes(),
        }
        assert identities[rel]["same_as_pretest"]
    start = json.loads((results / "settings-freeze-v1.test-started.json").read_bytes())
    assert datetime.fromisoformat(pin_time) < datetime.fromisoformat(start["recorded_at"]) < datetime.fromisoformat(report["recorded_at"])
    assert report["settings_freeze_sha256"] == sha(results / "settings-freeze-v1.json") == start["settings_freeze_sha256"]
    assert report["data_manifest_sha256"] == sha(root / "docs/lesson-products/data-v2-manifest.json")
    for name, digest in frozen["source_sha256s"].items():
        assert sha(root / "tiny_perceptron/lesson_products" / name) == digest
    for product in ("b", "c"):
        public = json.loads((results / f"{product}-artifact-manifest.json").read_bytes())
        assert {name: value["sha256"] for name, value in public["files"].items()} == frozen["products"][product]
        assert public["settings_freeze_sha256"] == report["settings_freeze_sha256"]

    # Regenerate only deterministic finite inputs, in memory. Hash their exact serialized bytes.
    b_rows, c_rows, config = build_rows()
    generated = {}
    for product, rows in (("b", b_rows), ("c", c_rows)):
        digest = hashlib.sha256()
        size = 0
        for row in rows:
            encoded = (canonical(row) + "\n").encode()
            digest.update(encoded)
            size += len(encoded)
        generated[product + ".jsonl"] = {"sha256": digest.hexdigest(), "bytes": size, "rows": len(rows)}
    audit = audit_rows(b_rows, c_rows)
    assert generated == manifest["files"]
    assert json.loads(json.dumps(audit)) == manifest["audit"] and config == manifest["config"]
    b_by_id = {row["id"]: row for row in b_rows}
    c_by_id = {row["id"]: row for row in c_rows}

    # Compute answers directly from raw logits, not from hand-entered report totals.
    counts = defaultdict(Counter)
    totals = Counter()
    raw_hash = hashlib.sha256()
    raw_bytes = 0
    raw_rows = 0
    wrong_saved_answers = 0
    failure_rows = {}
    for line in gzip.open(args.raw_gzip, "rb"):
        raw_hash.update(line)
        raw_bytes += len(line)
        raw_rows += 1
        row = json.loads(line)
        totals[(row["product"], row["split"], row.get("mode", ""))] += 1
        if row["product"] in ("b", "mechanism"):
            input_row = b_by_id[row["id"]]
            assert row["gold"] == input_row["gold"]
            scores = row.get("scores_by_depth", {"5": row.get("raw_scores")})
            for depth, values in scores.items():
                answer = max(range(len(values)), key=values.__getitem__)
                if "answers_by_depth" in row:
                    wrong_saved_answers += answer != row["answers_by_depth"][depth]
                group = f"{row['product']}/{row.get('mode', '')}/{row['split']}/{depth}/{row['task']}"
                counts[group]["count"] += 1
                counts[group]["correct"] += answer == row["gold"]
                if row["product"] == "b" and row["split"] == "test" and depth == "5" and input_row.get("strata", {}).get("repeated_swap"):
                    counts["repeated_swap"]["count"] += 1
                    counts["repeated_swap"]["correct"] += answer == row["gold"]
        elif row["product"] == "c":
            input_row = c_by_id[row["id"]]
            assert row["gold"] == input_row["gold"]
            selected = min(zip(row["candidates"], row["raw_scores"]), key=lambda pair: (-pair[1], pair[0]))[0]
            group = f"c/{row['split']}/{row['question']}"
            counts[group]["count"] += 1
            counts[group]["correct"] += selected == row["gold"]
            if row["split"] == "test":
                assert selected == row["selected"]
                if row["question"] == "choice":
                    counts["c_reorder"]["count"] += 1
                    counts["c_reorder"]["correct"] += selected == row["reverse_selected"]
            elif selected != row["gold"]:
                failure_rows[row["id"]] = row
    assert raw_hash.hexdigest() == report["predictions"]["sha256"]
    assert raw_bytes == report["predictions"]["bytes"]
    assert wrong_saved_answers == 0
    compare = {
        "b//test/5/boxes": report["b"]["5"]["boxes"],
        "b//test/5/sum": report["b"]["5"]["sum"],
        "repeated_swap": report["b"]["5"]["repeated_swap"],
        "c/test/choice": report["c"]["choice"]["after"],
        "c/test/info": report["c"]["info"]["after"],
        "c/test/rubric": report["c"]["rubric"]["after"],
        "c_reorder": {"count": report["c_reorder"]["count"], "correct": report["c_reorder"]["consistent"]},
    }
    for group, expected in compare.items():
        assert dict(counts[group]) == {key: expected[key] for key in ("count", "correct")}
    for depth in range(1, 7):
        for task in ("boxes", "sum"):
            assert dict(counts[f"b//test/{depth}/{task}"]) == {key: report["b"][str(depth)][task][key] for key in ("count", "correct")}
            for mode in ("fixed3", "odd"):
                assert dict(counts[f"mechanism/{mode}/test/{depth}/{task}"]) == {key: report["mechanisms"][mode]["evaluation"][str(depth)][task][key] for key in ("count", "correct")}
    for split in ("wording_shift", "entity_shift", "long_shift"):
        for task in ("boxes", "sum"):
            if task in report["shifts"][split]:
                assert dict(counts[f"b//{split}/5/{task}"]) == {key: report["shifts"][split][task][key] for key in ("count", "correct")}
    for question, values in report["shifts"]["c_wording"].items():
        assert dict(counts[f"c/wording_shift/{question}"]) == {key: values["after"][key] for key in ("count", "correct")}

    failures = json.loads((results / "observed-failure-examples-v1.json").read_bytes())
    assert failures["report_sha256"] == sha(results / "final-report-v1.json")
    assert failures["raw_predictions_sha256"] == raw_hash.hexdigest()
    failure_checks = []
    for example in failures["examples"]:
        raw = failure_rows[example["id"]]
        original = c_by_id[example["id"]]
        assert original["state"] == example["state"]
        assert raw["gold"] == example["gold"] and raw["raw_scores"] == example["raw_scores"]
        assert min(zip(raw["candidates"], raw["raw_scores"]), key=lambda pair: (-pair[1], pair[0]))[0] == example["predicted"] != example["gold"]
        failure_checks.append({"state": original["state"], "gold": raw["gold"], "predicted": example["predicted"], "matches_raw": True})

    moe = json.loads((root / "docs/selftrained/results/public-raw/moe/test/metrics.json").read_bytes())
    assert moe["evaluation_complete"] and not moe["limited_smoke"]
    ocr = moe["per_task_final_reply"]["ocr"]["exact"]
    assert ocr["numerator"] / ocr["denominator"] < moe["thresholds"]["ocr_exact"]
    cpu = json.loads((results / "public-cpu-acceptance-v1.json").read_bytes())
    assert cpu["status"] == "passed"
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-19-20-v1/verify_historical_scores.py --raw-gzip /workspace/work/reference/omni-technical-19-17-v1/raw/final-report-v1.predictions.jsonl.gz --output docs/technical-reviews/artifacts/omni-technical-19-20-v1/execution-v1.json",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "scope": "Read and reaggregate preserved primary records; deterministic input generation and logit argmax only. No original model inference, formal heldout evaluation, new training, GPU or paid service invocation.",
        "pretest_pin": pin,
        "pretest_commit_time": pin_time,
        "first_test_start": start["recorded_at"],
        "final_report_recorded_at": report["recorded_at"],
        "identities": identities,
        "raw": {"local_cache": str(args.raw_gzip), "official_url": "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/raw/final-report-v1.predictions.jsonl.gz", "gzip_sha256": sha(args.raw_gzip), "decompressed_sha256": raw_hash.hexdigest(), "decompressed_bytes": raw_bytes, "rows": raw_rows},
        "generated_input_files": generated,
        "data_audit": {"joint_family_overlap": audit["joint_family_overlap"], "b_shared_observations": audit["b"]["cross_split_shared_observations"], "c_shared_observations": audit["c"]["cross_split_shared_observations"], "manifest_config_and_audit_equal": True},
        "raw_groups": {"/".join(key): value for key, value in totals.items()},
        "recomputed_scores": {key: {**counts[key], "percentage": round(100 * counts[key]["correct"] / counts[key]["count"], 2)} for key in compare},
        "all_b_and_mechanism_depths_1_to_6_and_shift_counts_equal_report": True,
        "saved_answers_disagree_with_argmax": wrong_saved_answers,
        "predeclared_protocol": frozen["protocol"],
        "failure_checks": failure_checks,
        "moe_not_all_original_criteria": {"test_questions": moe["count"], "ocr_exact": ocr, "ocr_exact_threshold": moe["thresholds"]["ocr_exact"], "below_threshold": True},
        "historical_public_cpu_receipt": {"path": "docs/lesson-products/results/public-cpu-acceptance-v1.json", "sha256": sha(results / "public-cpu-acceptance-v1.json"), "status": cpu["status"], "started_at": cpu["started_at"], "scope": cpu["scope"]},
        "result": "All asserted identities, historical ordering, raw-score counts, subset counts, failure records and failed MoE criterion agree. Historical records are evidence of the disclosed runs, not proof of undisclosed out-of-band conduct or general model capability.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"result": result["result"], "recomputed_scores": result["recomputed_scores"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
