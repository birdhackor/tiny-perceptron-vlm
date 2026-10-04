"""Read original experiment records and bind relevant configurations and denominators."""

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tiny_perceptron.capstone import build_dataset, digest  # noqa: E402

ART = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_finish_19_11"


def main():
    result = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU"}}
    rows, manifest = build_dataset()
    public = json.loads((ART / f"{PREFIX}-public-main-data.json").read_text())
    assert public["manifest"] == manifest and public["splits"] == rows
    result["dataset"] = {
        "manifest": manifest,
        "public_data_sha256": hashlib.sha256((ART / f"{PREFIX}-public-main-data.json").read_bytes()).hexdigest(),
        "matches_current_generator": True,
    }
    stages = []
    previous_hash = None
    for name in ["pretrain", "sft", "joint", "preference"]:
        path = ROOT / f"docs/course-experiments/results/capstone_{name}.json"
        shutil.copyfile(path, ART / f"{PREFIX}-experiment-{name}.json")
        report = json.loads(path.read_text())
        original = report["results"]
        assert original["data_manifest"] == manifest
        assert original["parent_checkpoint_sha256"] == previous_hash
        previous_hash = original["inference_export"]["sha256"]
        validation_path = ROOT / f"docs/course-experiments/capstone-evidence/{original['stage']}/validation.json"
        shutil.copyfile(validation_path, ART / f"{PREFIX}-validation-{name}.json")
        validation = json.loads(validation_path.read_text())
        assert validation["count"] == len(validation["records"]) == 84
        assert validation["end_to_end_correct"] == sum(record["end_to_end_correct"] for record in validation["records"])
        assert validation["end_to_end_correct"] == original["validation_summary"]["end_to_end_correct"]
        stages.append(
            {
                "stage": original["stage"],
                "updates": original["steps"],
                "requested_steps": original["requested_steps"],
                "schedule_completed": original["schedule_completed"],
                "seed": report["seed"],
                "gpu": report["gpu"],
                "python": report["python_version"],
                "torch": report["torch_version"],
                "revision": report["revision"],
                "config": original["parameters"],
                "effective_tokens": original["effective_tokens"],
                "training_loop_seconds": original["seconds"],
                "experiment_seconds": report["elapsed_seconds"],
                "timing_scope": report["timing_scope"],
                "validation_count": 84,
                "validation_correct": validation["end_to_end_correct"],
                "original_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "parent_checkpoint_sha256": original["parent_checkpoint_sha256"],
                "inference_export": original["inference_export"],
                "objective": original["objective"],
                "original_batch_record_limitation": "Report does not preserve per-update GPU batch IDs. Current default batch size is 24; this audit does not invent historical batch traces.",
            }
        )
    result["stages"] = stages
    for name in ["deployment", "student"]:
        path = ROOT / f"docs/course-experiments/results/capstone_{name}.json"
        shutil.copyfile(path, ART / f"{PREFIX}-experiment-{name}.json")
        report = json.loads(path.read_text())
        result[name] = {
            "gpu": report["gpu"],
            "python": report["python_version"],
            "torch": report["torch_version"],
            "seed": report["seed"],
            "revision": report["revision"],
            "experiment_seconds": report["elapsed_seconds"],
            "timing_scope": report["timing_scope"],
            "configuration_and_result": report["results"],
        }
    expected = next(row for row in rows["test"] if row["user"] == "1+2等於多少？" and row["task"] == "calculator")
    path = ROOT / "docs/course-experiments/capstone-evidence/student/test-kd-ptq4.json"
    shutil.copyfile(path, ART / f"{PREFIX}-student-test-kd-ptq4.json")
    original = json.loads(path.read_text())
    record = next(record for record in original["records"] if record["id"] == expected["id"])
    assert record["action_trace"]["raw"] == "TOOL:calculator:2+9"
    assert record["runtime"]["result"] == "11" and record["final_trace"]["raw"] == "DIRECT:12"
    result["student_failure"] = {
        "row": expected,
        "record": record,
        "count": original["count"],
        "calculator_denominator": original["by_task"]["calculator"]["count"],
        "data_manifest_sha256": digest(manifest),
    }
    source = ROOT / "scripts/course_experiments/capstone_deployment.py"
    shutil.copyfile(source, ART / f"{PREFIX}-repo-scripts_course_experiments_capstone_deployment.py.txt")
    (ART / f"{PREFIX}-evidence-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Read original four-stage manifests, parent hashes, configurations, timings and all 84 validation records.")
    print(
        "Verified shared public data and original student KD int4 failure within 90 held-out rows/12 calculator rows."
    )


if __name__ == "__main__":
    main()
