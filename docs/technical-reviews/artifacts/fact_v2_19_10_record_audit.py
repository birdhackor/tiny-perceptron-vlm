"""Audit historical run records and parent fingerprints without GPU timing."""

import hashlib
import json
import platform
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_10_"


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    historical = read(f"docs/technical-reviews/artifacts/{PREFIX}weight_inventory.json")
    student = read("docs/course-experiments/results/capstone_student.json")
    deployment = read("docs/course-experiments/results/capstone_deployment.json")
    result = {
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
        "scope": "Read-only audit of original GPU record provenance and timer arithmetic; no GPU timings measured in this review",
        "timings": {
            "device": student["device"],
            "gpu": student["gpu"],
            "torch": student["torch_version"],
            "python": student["python_version"],
            "experiment_total": student["elapsed_seconds"],
            "timing_scope": student["timing_scope"],
            "branches": {},
        },
        "parents": {},
    }
    for mode in ["ce", "kd"]:
        branch = read(f"docs/course-experiments/capstone-evidence/student/{mode}/train-report.json")
        assert branch == student["results"]["branches"][mode]
        result["timings"]["branches"][mode] = {
            "seconds": branch["seconds"],
            "rounded_to_3": round(branch["seconds"], 3),
            "steps": branch["steps"],
            "effective_tokens": branch["effective_tokens"],
            "mode": branch["mode"],
            "objective": branch["objective"],
        }
    result["timings"]["experiment_total_rounded_to_3"] = round(student["elapsed_seconds"], 3)
    for name, folder, experiment, expected in [
        ("teacher", "v2-dpo", "capstone_preference", "dpo"),
        ("recommended", "v2-joint", "capstone_joint", "joint"),
    ]:
        path = next((ROOT / f"outputs/integration-runs/{folder}/capstone-review/{experiment}").glob("*/model.pt"))
        raw = path.read_bytes()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        tables = {}
        for key, tensor in payload["model"].items():
            hashed = hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()
            assert hashed == historical[expected]["restored_tensors"][key]["sha256"]
            tables[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": tensor.numel() * tensor.element_size(),
                "sha256": hashed,
            }
        result["parents"][name] = {
            "file_bytes": len(raw),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "config": payload["config"],
            "stage": payload["stage"],
            "step": payload["step"],
            "tensors": tables,
            "equals_deployment_export_tensors": True,
        }
    assert result["parents"]["teacher"]["file_sha256"] == student["results"]["teacher_checkpoint_sha256"]
    assert (
        result["parents"]["recommended"]["file_sha256"]
        == deployment["results"]["joint_ptq"]["joint-int4"]["source_checkpoint_sha256"]
    )
    result["selection_record"] = read("docs/course-experiments/capstone-selection.json")
    result["teacher_distinct_from_selected_joint"] = (
        result["parents"]["teacher"]["file_sha256"] != result["parents"]["recommended"]["file_sha256"]
    )
    (OUT / f"{PREFIX}gpu_record_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "timings": result["timings"],
                "parent_hashes": {k: v["file_sha256"] for k, v in result["parents"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
