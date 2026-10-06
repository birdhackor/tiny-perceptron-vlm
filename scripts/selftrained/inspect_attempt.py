"""Read bounded attempt evidence through the Volume client; never start an App."""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

VOLUME_NAME = "tiny-perceptron-course"
MAX_BYTES = 64 * 1024 * 1024
FILES = (
    "execution.json",
    "receipt.json",
    "runner.log",
    "train-receipt.json",
    "metrics.jsonl",
    "inference-manifest.json",
)
STAGES = ("pretrain", "sft", "vision", "ocr", "audio", "joint", "validation", "freeze", "test", "prepare", "release")
SAFE_EXPORT_FILES = ("model.safetensors", "model-config.json", "tokenizer.json", "inference-manifest.json")
TRAIN_STAGES = STAGES[:6]


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value) or ".." in value:
        raise ValueError("Attempt inspection requires plain versioned names")
    return value


def safe_export_gate(output, inspection, expected_receipt_sha256):
    observed = {item["path"]: item for item in inspection["files"] if item["status"] == "present"}
    if observed.get("receipt.json", {}).get("sha256") != expected_receipt_sha256:
        raise ValueError("Safe export requires the exact reviewed completed receipt SHA-256")
    source = json.loads((output / "receipt.json").read_text())
    if (
        source.get("status") != "completed"
        or source.get("batch_id") != inspection["batch_id"]
        or source.get("stage") != inspection["stage"]
        or source.get("run_id") != inspection["run_id"]
        or not re.fullmatch(r"[a-f0-9]{40}", source.get("revision", ""))
        or not re.fullmatch(r"[a-f0-9]{64}", source.get("manifest_sha256", ""))
    ):
        raise ValueError("Only the exact completed immutable training attempt can supply safe exports")
    files = {}
    for item in source.get("files", []):
        name = item.get("path")
        if (
            not isinstance(name, str)
            or name in files
            or type(item.get("bytes")) is not int
            or item["bytes"] < 0
            or not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", ""))
        ):
            raise ValueError("Completed receipt file identities are invalid or duplicated")
        files[name] = item
    for name in (*SAFE_EXPORT_FILES, "best.pt", "execution.json", "train-receipt.json"):
        if name not in files:
            raise ValueError("Completed receipt lacks its selected checkpoint or four safe export identities")
    for name, actual in observed.items():
        if name == "receipt.json":
            continue
        expected = files.get(name)
        if not expected or any(actual[key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError("Downloaded metadata differs from the pinned completed receipt")
    execution = json.loads((output / "execution.json").read_text())
    training = json.loads((output / "train-receipt.json").read_text())
    inference = json.loads((output / "inference-manifest.json").read_text())
    if (
        execution.get("status") != "completed"
        or any(
            execution.get(key) != source.get(key)
            for key in ("run_id", "batch_id", "stage", "revision", "manifest_sha256", "job")
        )
        or training.get("stage") != source["stage"]
        or training.get("architecture") != source["job"].get("architecture", "moe")
        or training.get("steps") != source["job"].get("steps", 300)
        or training.get("completed_requested_steps") is not True
        or training.get("selected_checkpoint_available") is not True
        or training.get("inference_exported") is not True
        or training.get("test_used_for_selection") is not False
        or training.get("origin", {}).get("kind") != "all-neural-weights-random"
        or inference.get("origin", {}).get("kind") != "all-neural-weights-random"
        or inference.get("stage") != source["stage"]
        or inference.get("selection") != "validation_loss"
        or inference.get("selected_checkpoint_sha256") != files["best.pt"]["sha256"]
        or inference.get("files") != {name: files[name]["sha256"] for name in SAFE_EXPORT_FILES[:3]}
    ):
        raise ValueError("Safe export lacks completed from-random selected-checkpoint provenance")
    pending_bytes = sum(files[name]["bytes"] for name in SAFE_EXPORT_FILES[:3])
    if inspection["total_downloaded_bytes"] + pending_bytes > MAX_BYTES:
        raise ValueError("Safe export exceeds the 64 MiB total read allowance")
    inspection.update(
        source_receipt_sha256=expected_receipt_sha256,
        source_revision=source["revision"],
        source_manifest_sha256=source["manifest_sha256"],
        selected_checkpoint_sha256=files["best.pt"]["sha256"],
    )
    return files


def inspect_attempt(batch_id, stage, run_id, output_dir, revision, safe_export=False, expected_receipt_sha256=None):
    import modal

    safe_name(batch_id)
    safe_name(run_id)
    if stage not in STAGES or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Inspection requires a known stage and exact observing source revision")
    if safe_export and (stage not in TRAIN_STAGES or not re.fullmatch(r"[a-f0-9]{64}", expected_receipt_sha256 or "")):
        raise ValueError("Safe export requires a training stage and exact reviewed receipt SHA-256")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    prefix = f"selftrained/{batch_id}/{stage}/{run_id}"
    receipt = {
        "schema": "selftrained-read-only-attempt-inspection-v1",
        "observed_at": datetime.now(UTC).isoformat(),
        "observing_revision": revision,
        "volume_name": VOLUME_NAME,
        "volume_relative_directory": prefix,
        "batch_id": batch_id,
        "stage": stage,
        "run_id": run_id,
        "status": "reading",
        "files": [],
        "total_downloaded_bytes": 0,
        "maximum_bytes": MAX_BYTES,
        "remote_container_started": False,
        "gpu_used": False,
        "ledger_written": False,
        "weights_read": False,
        "mode": "safe-export" if safe_export else "metadata",
    }
    try:
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

        def download(name):
            path = output / name
            checksum, size = hashlib.sha256(), 0
            try:
                with path.open("xb") as stream:
                    for chunk in volume.read_file(prefix + "/" + name):
                        if name == "model.safetensors" and chunk:
                            receipt["weights_read"] = True
                        receipt["total_downloaded_bytes"] += len(chunk)
                        if receipt["total_downloaded_bytes"] > MAX_BYTES:
                            raise ValueError("Attempt evidence exceeds the 64 MiB total read allowance")
                        stream.write(chunk)
                        checksum.update(chunk)
                        size += len(chunk)
            except FileNotFoundError:
                path.unlink(missing_ok=True)
                receipt["files"].append({"path": name, "status": "absent"})
            else:
                receipt["files"].append(
                    {"path": name, "status": "present", "bytes": size, "sha256": checksum.hexdigest()}
                )

        for name in FILES:
            download(name)
        if safe_export:
            expected_files = safe_export_gate(output, receipt, expected_receipt_sha256)
            for name in SAFE_EXPORT_FILES[:3]:
                download(name)
                actual = receipt["files"][-1]
                expected = expected_files[name]
                if actual["status"] != "present" or any(actual[key] != expected[key] for key in ("bytes", "sha256")):
                    raise ValueError("Downloaded safe export differs from the pinned completed receipt")
            config = json.loads((output / "model-config.json").read_text())
            training = json.loads((output / "train-receipt.json").read_text())
            if config != training.get("config") or config.get("architecture") != training.get("architecture"):
                raise ValueError("Safe model config differs from completed training")
            receipt["verified_safe_files"] = [expected_files[name] for name in SAFE_EXPORT_FILES]
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        temporary = output / "inspection-receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(output / "inspection-receipt.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--safe-export", action="store_true", help="Read only the four pinned selected inference files")
    parser.add_argument("--expected-receipt-sha256", help="Required completed receipt SHA-256 for safe-export mode")
    options = parser.parse_args()
    receipt = inspect_attempt(
        options.batch_id,
        options.stage,
        options.run_id,
        options.output_dir,
        options.revision,
        options.safe_export,
        options.expected_receipt_sha256,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "files": receipt["files"],
                "total_downloaded_bytes": receipt["total_downloaded_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
