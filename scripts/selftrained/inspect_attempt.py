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


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value) or ".." in value:
        raise ValueError("Attempt inspection requires plain versioned names")
    return value


def inspect_attempt(batch_id, stage, run_id, output_dir, revision):
    import modal

    safe_name(batch_id)
    safe_name(run_id)
    if stage not in STAGES or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Inspection requires a known stage and exact observing source revision")
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
    }
    try:
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
        for name in FILES:
            path = output / name
            checksum, size = hashlib.sha256(), 0
            try:
                with path.open("xb") as stream:
                    for chunk in volume.read_file(prefix + "/" + name):
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
    options = parser.parse_args()
    receipt = inspect_attempt(options.batch_id, options.stage, options.run_id, options.output_dir, options.revision)
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
