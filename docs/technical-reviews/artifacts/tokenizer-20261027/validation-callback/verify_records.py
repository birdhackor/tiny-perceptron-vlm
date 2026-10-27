"""獨立核對 validation 頁紀錄；不訓練、不下載模型、不重跑最後測試推論。"""

import hashlib
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path.cwd()
OUTPUT = ROOT / "docs/technical-reviews/artifacts/tokenizer-20261027/validation-callback/record-checks.json"


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def main():
    from tiny_perceptron.lesson_products.data import SPLITS, build_rows, canonical
    from tiny_perceptron.lesson_products.model import CandidateScorer, Config, LoopedSolver, Tokenizer
    from tiny_perceptron.lesson_products.training import metrics

    started = time.monotonic()
    result = {
        "command": "uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/validation-callback/verify_records.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "CPU"},
        "scope": "Recompute deterministic data identities/counts and arithmetic from original saved records. Historical GPU training and model inference are not repeated.",
    }
    freeze = read("docs/course-review-20261027-tokenizer/freeze-02.json")
    page = next(p for p in freeze["pages"] if p["page_id"] == "validation")
    assert sha(page["source"]) == sha(page["snapshot"]) == page["source_sha256"]
    result["page_identity"] = page

    v2 = read("docs/selftrained/results/v2-final-public-results.json")
    v2_result = {"data_scope": v2["data_scope"], "architectures": {}}
    for architecture, training_path, wanted_steps in [("moe", "moe-native", 4000), ("dense", "dense-weighted", 10000)]:
        raw_path = f"docs/selftrained/results/public-raw/{architecture}/test/metrics.json"
        raw = read(raw_path)
        receipt = read(f"docs/selftrained/results/training-raw/{training_path}/raw/train-receipt.json")
        frozen = read(f"docs/selftrained/results/public-raw/{architecture}/freeze/frozen.json")
        projection = v2["architectures"][architecture]["final_test"]
        for key in ("count", "evaluation_complete", "per_task_final_reply", "thresholds", "generation_budget"):
            assert raw[key] == projection[key], (architecture, key)
        assert raw["count"] == 3734 and raw["evaluation_complete"]
        assert sum(item["count"] for item in raw["per_task_final_reply"].values()) == 3734
        assert receipt["steps"] == wanted_steps and receipt["completed_requested_steps"]
        assert receipt["train_records"] == 28876 and receipt["validation_records"] == 2435
        assert not receipt["test_used_for_selection"]
        assert not raw["teacher_forcing_used_for_generation"]
        assert not v2["architectures"][architecture]["capability_acceptance_passed"]
        selected_index = read(f"docs/selftrained/results/v2-{architecture}-public-final-test-stage-index.json")
        assert selected_index["checkpoint"]["selected_step"] == 1000
        v2_result["architectures"][architecture] = {
            "raw_metrics_sha256": sha(raw_path),
            "count": raw["count"],
            "completed_steps": receipt["steps"],
            "selected_step": selected_index["checkpoint"]["selected_step"],
            "test_used_for_selection": receipt["test_used_for_selection"],
            "teacher_forcing_used_for_generation": raw["teacher_forcing_used_for_generation"],
            "frozen_protocol_keys": list(frozen),
            "per_task_final_reply": {
                k: {"count": x["count"], "exact": x.get("exact"), "tool_roundtrip": x.get("tool_roundtrip")}
                for k, x in raw["per_task_final_reply"].items()
            },
            "capability_acceptance_passed": v2["architectures"][architecture]["capability_acceptance_passed"],
        }
    result["v2"] = v2_result

    cpu = read("docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json")
    assert cpu["chat_invocation_count"] == 8 and cpu["history_append_helper_count"] == 1
    assert len(cpu["invocations"]) == 9 and all(x["returncode"] == 0 for x in cpu["invocations"])
    result["historical_cpu_smoke"] = {
        k: cpu[k]
        for k in ["chat_invocation_count", "history_append_helper_count", "known_validation_examples", "limits"]
    }

    local_runs = read(
        "docs/selftrained/infrastructure/v2-local-stage/independent-review/final-independent-actual-results.json"
    )
    assert len(local_runs) == 9 and all(x["actual_exit"] == x["expected_exit"] for x in local_runs)
    assert [x["saved_step"] for x in local_runs[:3]] == [1, 1, 2]
    assert all(not x["output_created"] for x in local_runs[3:])
    result["historical_local_wrapper"] = [
        {k: x[k] for k in ("name", "expected_exit", "actual_exit", "output_created")} for x in local_runs
    ]

    natural = read("docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/scored/scores.json")
    assert natural["selected_variant"] == read("docs/natural-assistant/v4/selection.json")["selected_variant"] == "base"
    base = natural["variants"]["base"]["correct_counts"]
    natural_variants = {}
    for name, variant in natural["variants"].items():
        if name == "base":
            continue
        counts = variant["correct_counts"]
        assert counts["photo_summary"] > base["photo_summary"] and counts["photo_fact"] > base["photo_fact"]
        assert counts["voice_typed_reference_chat"] < base["voice_typed_reference_chat"]
        assert counts["voice_actual_asr_chat"] < base["voice_actual_asr_chat"]
        assert not variant["all_generations_complete"]
        natural_variants[name] = {
            "correct_counts": counts,
            "all_generations_complete": variant["all_generations_complete"],
        }
    final_natural = read(
        "docs/natural-assistant/evidence/v4-runtime/evaluate-37221188153/blind-review/scored/scores.json"
    )
    assert final_natural["asr"]["by_task"]["speech_chat"]["count"] == 4
    result["natural_selection"] = {
        "base": base,
        "candidates": natural_variants,
        "selected_variant": "base",
        "final_correct_counts": final_natural["correct_counts"],
        "speech_chat_recordings": 4,
        "scope": "Saved manual/rubric scores and selection provenance, not a new judgment of raw photos or a new model evaluation.",
    }

    manifest = read("docs/lesson-products/data-v2-manifest.json")
    b, c, config = build_rows()
    assert config == manifest["config"]
    data_checks = {"config": config, "datasets": {}}
    family_splits = defaultdict(set)
    for product, rows in (("b", b), ("c", c)):
        h = hashlib.sha256()
        byte_count = 0
        observations = defaultdict(lambda: {"splits": set(), "rows": 0, "labels": set()})
        counts = Counter()
        example = defaultdict(set)
        for row in rows:
            raw = (canonical(row) + "\n").encode()
            h.update(raw)
            byte_count += len(raw)
            if row["split"] not in SPLITS:
                continue
            family_splits[row["family_id"]].add(row["split"])
            key = row["state"] if product == "b" else (row["state"], row["question"], tuple(sorted(row["candidates"])))
            observations[key]["splits"].add(row["split"])
            observations[key]["rows"] += 1
            observations[key]["labels"].add(row["gold"])
            counts[f"{row['split']}/{row['task'] if product == 'b' else row['question']}"] += 1
            if product == "c" and row["state"] == "卡片進度;申請編號=?":
                example[row["question"]].add((row["split"], row["gold"]))
        shared = [x for x in observations.values() if len(x["splits"]) > 1]
        measured = {
            "counts": dict(counts),
            "unique_observations": len(observations),
            "cross_split_shared_observations": len(shared),
            "rows_in_shared_observations": sum(x["rows"] for x in shared),
            "conflicting_labels": sum(len(x["labels"]) > 1 for x in observations.values()),
        }
        for key, value in measured.items():
            assert value == manifest["audit"][product][key], (product, key)
        assert {"sha256": h.hexdigest(), "bytes": byte_count, "rows": len(rows)} == manifest["files"][
            product + ".jsonl"
        ]
        data_checks["datasets"][product] = {
            **measured,
            "regenerated_bytes": byte_count,
            "regenerated_sha256": h.hexdigest(),
            "rows": len(rows),
            "generic_example": {key: sorted(x) for key, x in example.items()},
        }
    assert all(len(splits) == 1 for splits in family_splits.values())
    data_checks["joint_family_overlap"] = 0
    result["deterministic_data"] = data_checks
    del b, c

    tokenizer = Tokenizer()
    config = Config(vocab_size=len(tokenizer), width=64, heads=4, ff=128)
    parameters = {
        "b": sum(p.numel() for p in LoopedSolver(Config(vocab_size=len(tokenizer))).parameters()),
        "c": sum(p.numel() for p in CandidateScorer(config).parameters()),
    }
    assert parameters == {"b": 170003, "c": 84545}
    result["parameters"] = parameters

    training = {}
    for product, filename, steps in [
        ("b", "initial-b-execution.json", 1000),
        ("c", "initial-c-execution.json", 800),
        ("fixed3", "mechanism-fixed3-runner-result.json", 1500),
        ("odd", "mechanism-odd135-runner-result.json", 1500),
    ]:
        j = read("docs/lesson-products/results/" + filename)
        run = j.get("execution", j)
        receipt = run["training"]
        assert run["gpu_name"] == "NVIDIA L4"
        assert receipt["steps_this_attempt"] == receipt["last_step"] == steps
        assert receipt["resume_verification"]["exact_next_step"]
        training[product] = {
            "gpu": run["gpu_name"],
            "steps": steps,
            "parameters": receipt["parameters"],
            "initial_weights_sha256": receipt["initial_weights_sha256"],
            "resume_verification": receipt["resume_verification"],
        }
    assert training["fixed3"]["initial_weights_sha256"] == training["odd"]["initial_weights_sha256"]
    result["historical_product_training"] = training

    final = read("docs/lesson-products/results/final-report-v1.json")
    settings_path = "docs/lesson-products/results/settings-freeze-v1.json"
    settings = read(settings_path)
    package = read("docs/lesson-products/results/publication-package-v1.json")
    assert sha(settings_path) == final["settings_freeze_sha256"] == package["files"]["settings-freeze.json"]["sha256"]
    for product, entries in settings["products"].items():
        for name, digest in entries.items():
            assert digest == package["files"][product + "/" + name]["sha256"]
    numbers = {
        "b_boxes": final["b"]["5"]["boxes"],
        "b_sum": final["b"]["5"]["sum"],
        "b_repeated_swap": final["b"]["5"]["repeated_swap"],
        "b_entity_shift": final["shifts"]["entity_shift"]["boxes"],
        "c_choice": final["c"]["choice"],
        "c_reorder": final["c_reorder"],
        "c_wording": final["shifts"]["c_wording"],
        "gates": final["gates"],
    }
    for name, wanted in [
        ("b_boxes", (2928, 2956)),
        ("b_sum", (2, 42)),
        ("b_repeated_swap", (32, 42)),
        ("b_entity_shift", (1486, 2956)),
    ]:
        assert (numbers[name]["correct"], numbers[name]["count"]) == wanted
    assert (numbers["c_choice"]["after"]["correct"], numbers["c_choice"]["after"]["count"]) == (11797, 11797)
    assert (numbers["c_reorder"]["consistent"], numbers["c_reorder"]["count"]) == (11797, 11797)
    assert numbers["c_wording"]["choice"]["after"]["correct"] == 11493
    assert all(x["after"]["cross_entropy"] > x["before"]["cross_entropy"] for x in numbers["c_wording"].values())
    result["product_final"] = numbers
    commit = "821a2c717f05ad7b57e08e88cd965e92ff0c1200"
    shown = subprocess.run(["git", "show", commit + ":" + settings_path], capture_output=True)
    assert shown.returncode == 0 and hashlib.sha256(shown.stdout).hexdigest() == sha(settings_path)
    commit_time = subprocess.run(
        ["git", "show", "-s", "--format=%cI", commit], capture_output=True, text=True, check=True
    ).stdout.strip()
    started_marker = read("docs/lesson-products/results/settings-freeze-v1.test-started.json")
    assert commit_time < started_marker["recorded_at"]
    result["pretest_freeze"] = {
        "commit": commit,
        "commit_time": commit_time,
        "first_test_started": started_marker["recorded_at"],
        "settings_sha256": sha(settings_path),
        "all_published_product_files_match_freeze": True,
    }
    logits = torch.tensor([[1.0, 3.0], [2.0, 0.0]])
    base, scaled = metrics(logits, [1, 0], 1), metrics(logits, [1, 0], 0.25)
    assert base["correct"] == scaled["correct"] == 2 and scaled["cross_entropy"] < base["cross_entropy"]
    result["temperature_demo"] = {"logits": logits.tolist(), "gold": [1, 0], "T1": base, "T025": scaled}
    smoke = read("docs/gpu-smoke-result.json")
    assert (
        smoke["parameters"] == 31584
        and smoke["checks"]["finished_80_steps"]
        and smoke["checks"]["intermediate_40_steps"]
    )
    assert smoke["resume"]["max_weight_difference"] == 0 and smoke["preflight"]["checkpoint_repo_private"]
    result["historical_gpu_smoke"] = smoke
    result["elapsed_seconds"] = time.monotonic() - started
    result["result"] = (
        "All assertions passed. Historical records checked; no model training or final inference performed."
    )
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"result": result["result"], "elapsed_seconds": result["elapsed_seconds"], "output": str(OUTPUT)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
