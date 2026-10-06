"""Bounded random-initialized training on the existing shared course budget.

``python scripts/selftrained/modal_runner.py ledger`` reads committed Volume
bytes and live unit rates from the authenticated client without a remote
container. Actual work requires separate reserve and execute invocations.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COURSE = Path("/course")
LEDGER = COURSE / "budget.json"
EXPERIMENTS = COURSE / "selftrained"
TOTAL_CAP_USD = Decimal("55.00")
HISTORY_FLOOR_USD = Decimal("9.84")
MAX_JOB_USD = Decimal("1.00")
MAX_EGRESS_GIB = Decimal("0.25")
BUILD_STORAGE_ALLOWANCE_USD = Decimal("0.04")
CONTROL_SECONDS = 180
MAX_LEDGER_BYTES = 4 * 1024 * 1024
MAX_BACKUP_BYTES = 512 * 1024 * 1024
MAX_REVIEW_BYTES = 64 * 1024 * 1024
TRAIN_STAGES = ("pretrain", "sft", "vision", "ocr", "audio", "joint")
GPU_STAGES = (*TRAIN_STAGES, "validation", "freeze", "test")
STAGES = ("prepare", *GPU_STAGES, "release")
SPEC = {
    "prepare": {"cpu": 2, "memory_gib": 4, "seconds": 300, "gpu": False},
    "release": {"cpu": 1, "memory_gib": 2, "seconds": 600, "gpu": False},
    **{stage: {"cpu": 2, "memory_gib": 8, "seconds": 900, "gpu": True} for stage in GPU_STAGES},
}


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value) or ".." in value:
        raise ValueError("Need a plain versioned name without path components")
    return value


def relative_path(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Need a relative path without traversal")
    return path


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ledger_total(ledger):
    if not isinstance(ledger, dict) or not isinstance(ledger.get("reservations"), list):
        raise RuntimeError("Existing shared course ledger is missing; refusing a new ledger")
    amounts = [Decimal(str(item["reserved_usd"])) for item in ledger["reservations"]]
    reported = Decimal(str(ledger.get("reserved_total_usd", "0")))
    if any(not value.is_finite() or value < 0 for value in [reported, *amounts]):
        raise ValueError("Existing reservations must be finite nonnegative USD")
    total = max(reported, sum(amounts, Decimal("0")))
    if total < HISTORY_FLOOR_USD:
        raise RuntimeError("Ledger precedes confirmed $9.84 history; reconcile before running")
    if total > TOTAL_CAP_USD:
        raise RuntimeError(f"Shared conservative reservation total already exceeds authorized ${TOTAL_CAP_USD}")
    return total


def resource_spec(stage, wall_seconds=None):
    if stage not in STAGES:
        raise ValueError("Unknown bounded stage")
    spec = dict(SPEC[stage])
    seconds = spec["seconds"] if wall_seconds is None else wall_seconds
    if type(seconds) is not int or (spec["gpu"] and seconds not in (600, 900, 1800)):
        raise ValueError("GPU wall_seconds must be exactly 600, 900 or 1800")
    if not spec["gpu"] and seconds != spec["seconds"]:
        raise ValueError("CPU stage wall_seconds must match its fixed resource timeout")
    return spec | {"seconds": seconds}


def reservation_guard(stage, rates, wall_seconds=None):
    if stage not in STAGES:
        raise ValueError("Unknown bounded stage")
    required = ("cpu_hour_cost", "mem_gib_hour_cost", "egress_gib_cost")
    if SPEC[stage]["gpu"]:
        required += ("gpu_hour_cost_l4",)
    if any(key not in rates for key in required):
        raise ValueError("Live billing API lacks the previously verified hourly/GiB fields")
    values = {key: Decimal(str(rates[key])) for key in required}
    if any(not value.is_finite() or value < 0 for value in values.values()):
        raise ValueError("Live rates must be finite and nonnegative")
    spec = resource_spec(stage, wall_seconds)
    hourly = spec["cpu"] * values["cpu_hour_cost"] + spec["memory_gib"] * values["mem_gib_hour_cost"]
    if spec["gpu"]:
        hourly += values["gpu_hour_cost_l4"]
    compute = Decimal(spec["seconds"] + 2) * hourly / Decimal(3600)
    auxiliary = (
        Decimal(2 * CONTROL_SECONDS + 4) * (values["cpu_hour_cost"] / 4 + values["mem_gib_hour_cost"]) / Decimal(3600)
    )
    egress = MAX_EGRESS_GIB * values["egress_gib_cost"]
    amount = (compute + auxiliary + egress + BUILD_STORAGE_ALLOWANCE_USD).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )
    if amount > MAX_JOB_USD:
        raise RuntimeError(f"Live worst-case reservation ${amount} exceeds per-attempt ${MAX_JOB_USD}")
    return {
        "reserved_usd": str(amount),
        "bounded_compute_usd": str(compute),
        "auxiliary_compute_usd": str(auxiliary),
        "egress_allowance_usd": str(egress),
        "bounded_egress_gib": str(MAX_EGRESS_GIB),
        "build_and_storage_allowance_usd": str(BUILD_STORAGE_ALLOWANCE_USD),
        "resource_spec": spec,
        "input_rates": {key: str(value) for key, value in values.items()},
        "unit_contract": "USD per physical CPU/L4/GiB memory hour; USD per GiB egress",
        "limitations": "Build/storage allowance is conservative, not a perpetual storage cap or invoice; failed attempts retain full reservation",
    }


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("initialization") != "random":
        raise ValueError("Manifest must explicitly identify the random-initialized main line")
    package = manifest.get("package", {})
    if not re.fullmatch(r"[a-f0-9]{40}", package.get("revision", "")):
        raise ValueError("Package requires immutable Git LFS commit")
    if package.get("kind") != "git-lfs" or package.get("repo_id") != "birdhackor/tiny-perceptron-vlm":
        raise ValueError("Main-line package must use the existing Git LFS repository")
    package_path = relative_path(package.get("path", ""))
    if package_path.parts[:2] != ("assets", "training") or package_path.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("Package must be a tar.gz under assets/training")
    for key, maximum in (("bytes", 512 * 1024 * 1024), ("unpacked_bytes", 1024 * 1024 * 1024)):
        if type(package.get(key)) is not int or not 0 < package[key] <= maximum:
            raise ValueError("Package size exceeds the bounded data transport")
    if not re.fullmatch(r"[a-f0-9]{64}", package.get("sha256", "")):
        raise ValueError("Package needs exact SHA-256")
    if not isinstance(manifest.get("model_config"), dict) or not manifest.get("records"):
        raise ValueError("Manifest needs the actual model config and all fixed records")
    seen = set()
    for item in [*manifest["records"], *manifest.get("assets", [])]:
        path = relative_path(item["path"])
        if path.as_posix() in seen or type(item.get("bytes")) is not int or item["bytes"] <= 0:
            raise ValueError("Data files need unique paths and positive byte counts")
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
            raise ValueError("Every record/asset requires exact SHA-256")
        seen.add(path.as_posix())
    return manifest


def validate_descriptor(value, checkpoint=False):
    safe_name(value.get("run_id", ""))
    if value.get("stage") not in STAGES:
        raise ValueError("Artifact descriptor requires exact prior stage")
    path = relative_path(value.get("path", ""))
    if len(path.parts) != 1 or (checkpoint and path.name not in ("latest.pt", "best.pt")):
        raise ValueError("Descriptor must name one actual saved checkpoint or receipt")
    if not re.fullmatch(r"[a-f0-9]{64}", value.get("sha256", "")):
        raise ValueError("Prior artifacts require exact SHA-256")
    return value


def validate_job(job):
    stage = job.get("stage")
    if job.get("schema_version") != 1 or stage not in STAGES:
        raise ValueError("Job needs one existing bounded stage")
    if job.get("architecture", "moe") not in ("moe", "dense"):
        raise ValueError("Architecture must be moe or dense")
    for key, default, lower, upper in (
        ("steps", 300, 1, 20000),
        ("batch_size", 16, 1, 64),
        ("context", 512, 32, 512),
        ("seed", 20261006, 0, 2**31 - 1),
        ("eval_every", 100, 1, 1000),
        ("save_every", 100, 1, 1000),
        ("max_new_tokens", 128, 1, 128),
        ("limit", 10000, 1, 10000),
    ):
        value = job.get(key, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{key} exceeds the bounded runner")
    spec = resource_spec(stage, job.get("wall_seconds"))
    seconds = job.get("max_seconds", spec["seconds"] - 180)
    if type(seconds) is not int or not 1 <= seconds <= spec["seconds"] - 180:
        raise ValueError("Runtime must leave 180 seconds for checkpoint/cleanup")
    learning_rate = Decimal(str(job.get("learning_rate", "0.001")))
    if not learning_rate.is_finite() or not Decimal("0.00001") <= learning_rate <= Decimal("0.01"):
        raise ValueError("Learning rate exceeds the bounded training range")
    sampling = job.get("sampling_mode", "bucket")
    if sampling not in ("bucket", "task-family") or (sampling != "bucket" and stage != "joint"):
        raise ValueError("task-family sampling is available only for joint training")
    freeze = job.get("freeze_perception_backbones", False)
    if type(freeze) is not bool or (freeze and stage != "joint"):
        raise ValueError("freeze_perception_backbones must be boolean and is available only for joint training")
    for key in ("tool_loss_weight", "numeric_run_loss_weight"):
        if key in job:
            value = job[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 1 or stage != "joint":
                raise ValueError(f"{key} must be a finite number >= 1 and is available only for joint training")
    if job.get("resume") and job.get("init_checkpoint"):
        raise ValueError("Resume and fresh-stage initialization are mutually exclusive")
    for field in ("resume", "init_checkpoint", "checkpoint"):
        if job.get(field):
            validate_descriptor(job[field], checkpoint=True)
    if stage in ("validation", "freeze", "test") and not job.get("checkpoint"):
        raise ValueError("Evaluation needs an exact checkpoint descriptor")
    if "public_export" in job:
        from scripts.selftrained.hf_transport import validate_public_evaluation

        if stage not in ("validation", "freeze", "test") or job.get("resume") or job.get("init_checkpoint"):
            raise ValueError("Public safe exports are available only for evaluation")
        if job["checkpoint"]["stage"] != "joint" or job["checkpoint"]["path"] != "best.pt":
            raise ValueError("Public evaluation must bind the selected private joint checkpoint")
        validate_public_evaluation(job["public_export"], job.get("architecture", "moe"))
    if stage in TRAIN_STAGES and stage != "pretrain" and not (job.get("resume") or job.get("init_checkpoint")):
        raise ValueError("Later stages must inherit the shared tokenizer and from-zero core")
    if stage == "test":
        validate_descriptor(job.get("protocol", {}))
        if job["protocol"]["stage"] != "freeze" or job["protocol"]["path"] != "frozen.json":
            raise ValueError("Final test requires the actual completed validation freeze protocol")
        if "limit" in job:
            raise ValueError("Frozen final test cannot limit its records")
    if job.get("resume_evaluation"):
        validate_descriptor(job["resume_evaluation"])
        if (
            stage not in ("validation", "freeze", "test")
            or job["resume_evaluation"]["path"] != "outputs.jsonl"
            or job["resume_evaluation"]["stage"] not in (("validation", "freeze") if stage == "freeze" else (stage,))
        ):
            raise ValueError("Evaluation resume must name the exact prior same-split raw outputs")
        if not re.fullmatch(r"[a-f0-9]{64}", job["resume_evaluation"].get("receipt_sha256", "")):
            raise ValueError("Evaluation resume also needs exact evaluation-receipt.json SHA")
    if stage == "freeze" and "limit" in job:
        raise ValueError("Final validation freeze must evaluate its entire fixed split")
    if stage == "release":
        from scripts.selftrained.hf_transport import validate_batch_release

        for export in validate_batch_release(job.get("release", {})):
            validate_descriptor(export["source"], checkpoint=True)
    return job


def committed_json(relative, revision):
    path = relative_path(relative)
    if path.parts[:2] != ("docs", "selftrained") or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Job/manifest must be committed under docs/selftrained at a full Git SHA")
    committed = subprocess.run(
        ["git", "show", f"{revision}:{path.as_posix()}"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    if (ROOT / path).read_bytes() != committed:
        raise ValueError("Input differs from selected committed Git revision")
    return json.loads(committed), hashlib.sha256(committed).hexdigest()


def source_gate(revision):
    paths = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "scripts/selftrained",
            "tiny_perceptron",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    required = {
        "scripts/selftrained/modal_runner.py",
        "scripts/selftrained/hf_transport.py",
        "scripts/selftrained/train.py",
        "scripts/selftrained/evaluate.py",
    }
    if not required <= set(paths):
        raise ValueError("Actual model/trainer/evaluator must be committed before reserving work")
    for path in paths:
        content = subprocess.run(
            ["git", "show", f"{revision}:{path}"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        if (ROOT / path).read_bytes() != content:
            raise ValueError(f"Source differs from selected commit: {path}")
    return {path: sha256(ROOT / path) for path in paths}


def live_snapshot():
    import modal

    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "rates": {key: str(value) for key, value in modal.Workspace.from_context().billing.rates().items()},
        "scope": "Live workspace unit rates; reservation is not an invoice or project billing delta",
    }


def read_live_ledger():
    import modal

    chunks, total = [], 0
    shared = modal.Volume.from_name("tiny-perceptron-course", create_if_missing=False)
    for chunk in shared.read_file("budget.json"):
        total += len(chunk)
        if total > MAX_LEDGER_BYTES:
            raise ValueError("Budget ledger exceeds bounded review size")
        chunks.append(chunk)
    raw = b"".join(chunks)
    ledger = json.loads(raw)
    reserved = ledger_total(ledger)
    live = live_snapshot()
    return {
        "observed_at": live["observed_at"],
        "volume_name": "tiny-perceptron-course",
        "ledger_path": "/course/budget.json",
        "ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "ledger": ledger,
        "authorized_cap_usd": str(TOTAL_CAP_USD),
        "reserved_total_usd": str(reserved),
        "remaining_reserved_budget_usd": str(TOTAL_CAP_USD - reserved),
        "live": live,
        "stage_guards": {stage: reservation_guard(stage, live["rates"]) for stage in STAGES},
        "remote_container_started": False,
        "gpu_used": False,
        "ledger_written": False,
        "scope": "Latest committed shared Volume bytes; conservative reservations are not measured invoices",
    }


def reserve_entry(ledger, run_id, batch_id, revision, manifest_sha, job, job_sha, live, source_hashes):
    prior = ledger_total(ledger)
    if any(entry.get("run_id") == run_id for entry in ledger["reservations"]):
        raise ValueError("Each attempt needs a fresh run ID; prior reservations are never reused")
    numerical_job = {key: value for key, value in job.items() if key not in ("wall_seconds", "max_seconds")}
    logical_sha = canonical_sha({"batch_id": batch_id, "manifest_sha256": manifest_sha, "job": numerical_job})
    if any(
        entry.get("logical_job_sha256") == logical_sha and entry.get("status") == "completed"
        for entry in ledger["reservations"]
    ):
        raise ValueError("This exact job already completed; do not rerun successful numbers")
    guard = reservation_guard(job["stage"], live["rates"], job.get("wall_seconds"))
    amount = Decimal(guard["reserved_usd"])
    if prior + amount > TOTAL_CAP_USD:
        raise RuntimeError(f"Cumulative reservation {prior} + {amount} exceeds authorized ${TOTAL_CAP_USD}")
    entry = {
        "run_id": run_id,
        "batch_id": batch_id,
        "experiment_id": f"selftrained_{job['stage']}",
        "stage": job["stage"],
        "revision": revision,
        "manifest_sha256": manifest_sha,
        "job_sha256": job_sha,
        "logical_job_sha256": logical_sha,
        "runtime_options": job,
        "source_sha256": source_hashes,
        "reserved_usd": str(amount),
        "status": "reserved",
        "billing_before": live,
        "compute_guard": guard,
        "scope": "Existing original $10 + authorized $30 + $15 course ledger; all history and failed reservations retained",
    }
    ledger["reservations"].append(entry)
    ledger.update(budget_usd=str(TOTAL_CAP_USD), reserved_total_usd=str(prior + amount))
    return entry


def data_root(manifest_sha):
    return EXPERIMENTS / "datasets" / manifest_sha


def artifact_path(batch_id, descriptor):
    validate_descriptor(descriptor, checkpoint=descriptor["path"] in ("latest.pt", "best.pt"))
    path = (
        EXPERIMENTS / safe_name(batch_id) / descriptor["stage"] / safe_name(descriptor["run_id"]) / descriptor["path"]
    )
    if path.is_symlink() or not path.is_file() or sha256(path) != descriptor["sha256"]:
        raise ValueError("Prior Volume artifact differs from its exact checkpoint/protocol SHA")
    return path


def prepared_records_gate(manifest, manifest_sha):
    """Bounded control check; full asset verification still precedes training."""
    from scripts.selftrained.hf_transport import verify_file

    root = data_root(manifest_sha)
    receipt_path = root / "prepare-receipt.json"
    if not receipt_path.is_file():
        raise ValueError("Exact frozen package must complete CPU prepare before GPU reservation")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "completed"
        or receipt.get("manifest_sha256") != manifest_sha
        or receipt.get("package") != manifest["package"]
        or receipt.get("records") != manifest["records"]
        or receipt.get("asset_count") != len(manifest.get("assets", []))
    ):
        raise ValueError("Prepared package/records identity differs from the exact frozen manifest")
    for item in manifest["records"]:
        verify_file(root, item)
    return root


def readiness(manifest, manifest_sha):
    from scripts.selftrained.hf_transport import verify_file

    root = prepared_records_gate(manifest, manifest_sha)
    for item in manifest.get("assets", []):
        verify_file(root, item)
    return root


def local_readiness(manifest, manifest_sha, archive="/package/dataset.tar.gz", destination=None):
    """Verify the exact prepared archive on local disk, avoiding per-file Volume I/O."""
    from scripts.selftrained.hf_transport import unpack_verified_archive, verify_file

    root = Path(destination) if destination is not None else Path("/tmp/selftrained-data") / manifest_sha
    started = time.monotonic()
    print(json.dumps({"event": "local_archive_verification_started", "manifest_sha256": manifest_sha}), flush=True)
    unpack_verified_archive(archive, root, manifest["package"])
    unpacked = time.monotonic()
    for item in [*manifest["records"], *manifest.get("assets", [])]:
        verify_file(root, item)
    finished = time.monotonic()
    timing = {
        "status": "complete",
        "storage": "ephemeral-local-verified-git-lfs-archive",
        "root": str(root),
        "manifest_sha256": manifest_sha,
        "archive_sha256": manifest["package"]["sha256"],
        "archive_bytes": manifest["package"]["bytes"],
        "record_files_verified": len(manifest["records"]),
        "asset_files_verified": len(manifest.get("assets", [])),
        "archive_verify_unpack_seconds": unpacked - started,
        "all_file_hash_seconds": finished - unpacked,
        "total_seconds": finished - started,
    }
    print(json.dumps({"event": "local_data_readiness_complete", **timing}), flush=True)
    return root, timing


def artifact_gate(batch_id, descriptor, manifest_sha, architecture=None):
    path = artifact_path(batch_id, descriptor)
    execution_path = path.parent / "execution.json"
    if not execution_path.is_file():
        raise ValueError("Prior artifact is missing its actual execution provenance")
    execution = json.loads(execution_path.read_text())
    if execution.get("manifest_sha256") != manifest_sha:
        raise ValueError("Prior checkpoint/protocol belongs to a different frozen data/config manifest")
    if architecture and execution.get("job", {}).get("architecture", "moe") != architecture:
        raise ValueError("Dense and MoE cannot share checkpoints")
    return path, execution


def release_sources_gate(batch_id, release, manifest_sha, manifest):
    from scripts.selftrained.hf_transport import approved_batch_exports, validate_batch_release

    sources = {}
    for export in validate_batch_release(release):
        path, execution = artifact_gate(batch_id, export["source"], manifest_sha, export["architecture"])
        sources[export["name"]] = {"directory": path.parent, "execution": execution}
    approved_batch_exports(sources, release, manifest_sha, manifest)
    return sources


def public_evaluation_source_gate(batch_id, job, manifest_sha, manifest):
    """Bind public file pins to a completed private source before reserving.

    Only the trusted checkpoint and small provenance JSON are read here; public
    weights are downloaded anonymously and verified on the executing GPU.
    """
    from scripts.selftrained.hf_transport import (
        public_evaluation_identity,
        validate_public_evaluation,
        verify_file,
    )

    architecture = job.get("architecture", "moe")
    files = validate_public_evaluation(job["public_export"], architecture)
    if batch_id != "selftrained-v2":
        raise ValueError("The authorized V2 public export requires its V2 private lineage")
    checkpoint = job["checkpoint"]
    path, execution = artifact_gate(batch_id, checkpoint, manifest_sha, architecture)
    receipt = json.loads((path.parent / "receipt.json").read_text())
    if (
        checkpoint["stage"] != "joint"
        or checkpoint["path"] != "best.pt"
        or execution.get("status") != "completed"
        or any(
            receipt.get(key) != execution.get(key)
            for key in ("status", "revision", "stage", "run_id", "manifest_sha256", "job")
        )
        or receipt.get("stage") != checkpoint["stage"]
        or receipt.get("run_id") != checkpoint["run_id"]
    ):
        raise ValueError("Public evaluation source must have its completed exact private execution/receipt")
    recorded = receipt.get("files", [])
    entries = {item["path"]: item for item in recorded}
    if len(entries) != len(recorded) or entries.get("best.pt", {}).get("sha256") != checkpoint["sha256"]:
        raise ValueError("Public evaluation source receipt differs from its exact selected checkpoint")
    if "train-receipt.json" not in entries:
        raise ValueError("Public evaluation source is missing its recorded training receipt")
    if any(
        {key: entries.get(name, {}).get(key) for key in ("path", "sha256", "bytes")} != item
        for name, item in files.items()
    ):
        raise ValueError("Public file pins differ from the actual completed private safe export")
    training = json.loads(verify_file(path.parent, entries["train-receipt.json"]).read_text())
    if training.get("steps", -1) < execution.get("job", {}).get("steps", 0):
        raise ValueError("Public evaluation source did not finish its requested training steps")
    return public_evaluation_identity(path.parent, job["public_export"], checkpoint, manifest, architecture, training)


def evaluation_weight_identity(job):
    if "public_export" in job:
        from scripts.selftrained.hf_transport import validate_public_evaluation

        files = validate_public_evaluation(job["public_export"], job.get("architecture", "moe"))
        return {
            "weight_source": "safe_export",
            "checkpoint_sha256": files["model.safetensors"]["sha256"],
            "safe_weights_sha256": files["model.safetensors"]["sha256"],
            "inference_manifest_sha256": files["inference-manifest.json"]["sha256"],
            "selected_checkpoint_sha256": job["checkpoint"]["sha256"],
        }
    return {
        "weight_source": "private_checkpoint",
        "checkpoint_sha256": job["checkpoint"]["sha256"],
        "safe_weights_sha256": None,
        "inference_manifest_sha256": None,
        "selected_checkpoint_sha256": None,
    }


def evaluation_public_lineage_gate(execution, job):
    if execution.get("job", {}).get("public_export") != job.get("public_export"):
        raise ValueError(
            "Evaluation continuation/protocol must bind the same immutable public descriptor and weight source"
        )


def evaluation_resume_gate(path, job, protocol_sha=None):
    receipt_path = path.with_name("evaluation-receipt.json")
    expected_sha = job["resume_evaluation"]["receipt_sha256"]
    if not receipt_path.is_file() or sha256(receipt_path) != expected_sha:
        raise ValueError("Evaluation resume receipt differs from its pinned bytes")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "selftrained-evaluation-journal-v1"
        or receipt.get("split") != ("test" if job["stage"] == "test" else "validation")
        or receipt.get("checkpoint_sha256") != evaluation_weight_identity(job)["checkpoint_sha256"]
        or receipt.get("protocol_sha256") != protocol_sha
        or type(receipt.get("completed_count")) is not int
        or type(receipt.get("expected_count")) is not int
        or not 0 <= receipt["completed_count"] <= receipt["expected_count"]
        or (job["stage"] != "freeze" and receipt["completed_count"] == receipt["expected_count"])
        or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("conditions_sha256", ""))
    ):
        raise ValueError("Only unfinished rows under the same frozen checkpoint/protocol can resume")
    raw = path.read_bytes()
    size = receipt.get("output_byte_count")
    if type(size) is not int or not 0 <= size <= len(raw):
        raise ValueError("Evaluation receipt has invalid committed-prefix byte count")
    prefix = raw[:size]
    if hashlib.sha256(prefix).hexdigest() != receipt.get("outputs_sha256") or (prefix and not prefix.endswith(b"\n")):
        raise ValueError("Evaluation committed raw prefix differs from actual bytes")
    rows = [json.loads(line) for line in prefix.splitlines()]
    ids = [row["record"]["id"] for row in rows]
    if len(ids) != receipt["completed_count"] or len(ids) != len(set(ids)):
        raise ValueError("Completed evaluation IDs are inconsistent or duplicated")
    if hashlib.sha256("".join(value + "\n" for value in ids).encode()).hexdigest() != receipt.get(
        "completed_ids_sha256"
    ):
        raise ValueError("Completed evaluation ID digest differs from actual raw prefix")
    return receipt


def test_protocol_gate(path, job, manifest, source_hashes, resume_path=None):
    protocol = json.loads(path.read_text())
    if path.with_suffix(path.suffix + ".test-started.json").exists() and resume_path is None:
        raise ValueError("This frozen final test already started; failed attempts are not silently retried")
    expected_data = {Path(item["path"]).name: item["sha256"] for item in manifest["records"]}
    if (
        protocol.get("version") != "selftrained-generation-v2"
        or protocol.get("generation_budget_policy") != "full-history-ceiling-min-remaining-context-v1"
        or protocol.get("test_once") is not True
        or protocol.get("weight_source", "private_checkpoint") != evaluation_weight_identity(job)["weight_source"]
        or any(
            protocol.get(key) != value
            for key, value in evaluation_weight_identity(job).items()
            if key != "weight_source"
        )
        or protocol.get("architecture") != job.get("architecture", "moe")
        or protocol.get("max_new_tokens") != job.get("max_new_tokens", 128)
        or protocol.get("data_sha256") != expected_data
        or protocol.get("controls") != "all"
        or any(source_hashes.get(name) != value for name, value in protocol.get("code_sha256", {}).items())
        or not protocol.get("code_sha256")
    ):
        raise ValueError("Final test differs from its frozen checkpoint/data/code/generation protocol")
    if resume_path is not None:
        evaluation_resume_gate(resume_path, job, sha256(path))
    return protocol


def trainer_command(job, manifest, root, output, batch_id, public_model_dir=None):
    script = "train.py" if job["stage"] in TRAIN_STAGES else "evaluate.py"
    command = [sys.executable, f"/repo/scripts/selftrained/{script}"]
    for record in manifest["records"]:
        command += ["--records", str(root / relative_path(record["path"]))]
    command += ["--asset-dir", str(root), "--output-dir", str(output), "--device", "cuda", "--threads", "2"]
    if job["stage"] in TRAIN_STAGES:
        config = output / "model-config.json"
        write_json(config, manifest["model_config"])
        command += [
            "--config",
            str(config),
            "--architecture",
            job.get("architecture", "moe"),
            "--stage",
            job["stage"],
            "--export-inference",
        ]
        for flag, default in (
            ("steps", 300),
            ("batch_size", 16),
            ("context", 512),
            ("learning_rate", "0.001"),
            ("seed", 20261006),
            ("eval_every", 100),
            ("save_every", 100),
        ):
            command += ["--" + flag.replace("_", "-"), str(job.get(flag, default))]
        if job.get("sampling_mode", "bucket") != "bucket":
            command += ["--sampling-mode", job["sampling_mode"]]
        if job.get("freeze_perception_backbones", False):
            command += ["--freeze-perception-backbones"]
        for flag in ("tool_loss_weight", "numeric_run_loss_weight"):
            if flag in job:
                command += ["--" + flag.replace("_", "-"), str(job[flag])]
        for field in ("resume", "init_checkpoint"):
            if job.get(field):
                command += ["--" + field.replace("_", "-"), str(artifact_path(batch_id, job[field]))]
    else:
        if "public_export" in job:
            if public_model_dir is None:
                raise ValueError("Public evaluation command requires its verified safe model directory")
            command += ["--model-dir", str(public_model_dir)]
        else:
            command += ["--checkpoint", str(artifact_path(batch_id, job["checkpoint"]))]
        command += [
            "--split",
            "test" if job["stage"] == "test" else "validation",
        ]
        command += ["--max-new-tokens", str(job.get("max_new_tokens", 128))]
        if "limit" in job:
            command += ["--limit", str(job["limit"])]
        if job["stage"] == "freeze":
            command += ["--freeze-protocol", str(output / "frozen.json")]
        if job["stage"] == "test":
            command += ["--protocol", str(artifact_path(batch_id, job["protocol"]))]
        if job.get("resume_evaluation"):
            command += ["--resume-output", str(artifact_path(batch_id, job["resume_evaluation"]))]
    return command


def finish_entry(ledger, run_id, status, receipt=None):
    entry = next((item for item in ledger["reservations"] if item.get("run_id") == run_id), None)
    if not entry or entry.get("experiment_id", "").startswith("selftrained_") is False:
        raise ValueError("Cannot finish an absent or unrelated reservation")
    if entry.get("status") == "completed" and status != "completed":
        return entry
    entry.update(status=status, finished_at=datetime.now(UTC).isoformat())
    if receipt:
        entry["receipt_sha256"] = canonical_sha(receipt)
    ledger_total(ledger)
    return entry


def register_modal():
    import modal

    phase = os.environ.get("SELFTRAINED_MODAL_PHASE", "control")
    app = modal.App("tiny-perceptron-selftrained")
    volume = modal.Volume.from_name("tiny-perceptron-course", create_if_missing=False)
    control = (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install("huggingface-hub==1.33.0")
        .env({"PYTHONPATH": "/repo:/repo/scripts/selftrained"})
        .add_local_dir(ROOT / "scripts/selftrained", "/repo/scripts/selftrained", ignore=["**/__pycache__/**"])
    )
    image, prepare_image = control, control
    if phase in ("prepare", "execute"):
        asset = Path(os.environ["SELFTRAINED_ASSET_FILE"])
        if not asset.is_file() or asset.suffixes[-2:] != [".tar", ".gz"]:
            raise ValueError("Execution requires the exact already verified Git LFS package")
    if phase == "prepare":
        prepare_image = control.add_local_file(asset, "/package/dataset.tar.gz")
    if phase == "execute":
        image = (
            modal.Image.debian_slim(python_version="3.13")
            .uv_sync(str(ROOT), extras=["cu126", "selftrained"], uv_version="0.12.22", extra_options="--no-dev")
            .env(
                {
                    "PYTHONPATH": "/repo:/repo/scripts/selftrained",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                }
            )
            .workdir("/repo")
            .add_local_dir(ROOT / "tiny_perceptron", "/repo/tiny_perceptron", ignore=["**/__pycache__/**"])
            .add_local_dir(ROOT / "scripts/selftrained", "/repo/scripts/selftrained", ignore=["**/__pycache__/**"])
            .add_local_file(asset, "/package/dataset.tar.gz")
        )

    @app.function(
        serialized=True,
        image=control,
        cpu=(0.25, 0.25),
        memory=(1024, 1024),
        volumes={"/course": volume},
        timeout=CONTROL_SECONDS,
        retries=0,
        max_containers=1,
        scaledown_window=2,
    )
    def reserve_remote(
        run_id, batch_id, revision, manifest, manifest_sha, job, job_sha, live, source_hashes, previous_sha
    ):
        volume.reload()
        if not LEDGER.is_file() or sha256(LEDGER) != previous_sha:
            raise RuntimeError("Live shared ledger changed or disappeared since client probe; reread before reserving")
        if job["stage"] not in ("prepare", "release"):
            prepared_records_gate(manifest, manifest_sha)
        if "public_export" in job:
            public_evaluation_source_gate(batch_id, job, manifest_sha, manifest)
        resume_path = None
        if job.get("resume_evaluation"):
            resume_path, resume_execution = artifact_gate(
                batch_id, job["resume_evaluation"], manifest_sha, job.get("architecture", "moe")
            )
            evaluation_public_lineage_gate(resume_execution, job)
            if job["stage"] in ("validation", "freeze"):
                evaluation_resume_gate(resume_path, job)
        for field in ("resume", "init_checkpoint", "checkpoint", "protocol"):
            if job.get(field):
                if field == "checkpoint" and "public_export" in job:
                    continue  # Already verified by the completed public source gate.
                path, source_execution = artifact_gate(
                    batch_id, job[field], manifest_sha, job.get("architecture", "moe")
                )
                if field == "protocol":
                    evaluation_public_lineage_gate(source_execution, job)
                    test_protocol_gate(path, job, manifest, source_hashes, resume_path)
        if job["stage"] == "release":
            release_sources_gate(batch_id, job["release"], manifest_sha, manifest)
        ledger = json.loads(LEDGER.read_text())
        entry = reserve_entry(ledger, run_id, batch_id, revision, manifest_sha, job, job_sha, live, source_hashes)
        write_json(LEDGER, ledger)
        volume.commit()
        return {"entry": entry, "budget_usd": ledger["budget_usd"], "reserved_total_usd": ledger["reserved_total_usd"]}

    @app.function(
        serialized=True,
        image=control,
        cpu=(0.25, 0.25),
        memory=(1024, 1024),
        volumes={"/course": volume},
        timeout=CONTROL_SECONDS,
        retries=0,
        max_containers=1,
        scaledown_window=2,
    )
    def finish_remote(run_id, status):
        volume.reload()
        ledger = json.loads(LEDGER.read_text())
        entry = finish_entry(ledger, run_id, status)
        write_json(LEDGER, ledger)
        volume.commit()
        return {"entry": entry, "reserved_total_usd": str(ledger_total(ledger))}

    def execute(stage, run_id, batch_id, revision, manifest, manifest_sha, job, job_sha):
        volume.reload()
        ledger = json.loads(LEDGER.read_text())
        entry = next((item for item in ledger["reservations"] if item.get("run_id") == run_id), None)
        expected = {
            "stage": stage,
            "revision": revision,
            "manifest_sha256": manifest_sha,
            "job_sha256": job_sha,
            "batch_id": batch_id,
            "status": "reserved",
        }
        if entry is None or any(entry.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Execution differs from exact prior shared-ledger reservation")
        entry["status"] = "running"
        write_json(LEDGER, ledger)
        volume.commit()
        output = EXPERIMENTS / batch_id / stage / run_id
        if output.exists():
            raise ValueError("Attempt output already exists; never overwrite evidence")
        output.mkdir(parents=True)
        receipt = {
            "run_id": run_id,
            "batch_id": batch_id,
            "stage": stage,
            "revision": revision,
            "manifest_sha256": manifest_sha,
            "job_sha256": job_sha,
            "job": job,
            "started_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "resource_spec": resource_spec(stage, job.get("wall_seconds")),
        }
        write_json(output / "execution.json", receipt)
        volume.commit()
        started, stop, error = time.monotonic(), threading.Event(), None

        def persist():
            while not stop.wait(30):
                volume.commit()

        thread = threading.Thread(target=persist, daemon=True)
        thread.start()
        try:
            if stage == "prepare":
                from scripts.selftrained.hf_transport import unpack_verified_archive, verify_file

                root = data_root(manifest_sha)
                if (root / "prepare-receipt.json").exists():
                    raise ValueError("Exact dataset already prepared; reuse its readiness receipt")
                unpack_verified_archive("/package/dataset.tar.gz", root, manifest["package"])
                for item in [*manifest["records"], *manifest.get("assets", [])]:
                    verify_file(root, item)
                write_json(
                    root / "prepare-receipt.json",
                    {
                        "status": "completed",
                        "manifest_sha256": manifest_sha,
                        "package": manifest["package"],
                        "records": manifest["records"],
                        "asset_count": len(manifest.get("assets", [])),
                        "revision": revision,
                    },
                )
            elif stage == "release":
                from scripts.selftrained.hf_transport import publish_batch_inference

                sources = release_sources_gate(batch_id, job["release"], manifest_sha, manifest)
                result = publish_batch_inference(
                    sources,
                    job["release"],
                    manifest_sha,
                    manifest,
                    os.environ["HF_TOKEN"],
                    output / "public-staging",
                )
                write_json(output / "hf-receipt.json", result)
            else:
                receipt["data_readiness"] = {
                    "status": "verifying",
                    "storage": "ephemeral-local-verified-git-lfs-archive",
                    "archive_sha256": manifest["package"]["sha256"],
                }
                write_json(output / "execution.json", receipt)
                volume.commit()
                root, receipt["data_readiness"] = local_readiness(manifest, manifest_sha)
                public_model_dir = None
                if "public_export" in job:
                    from scripts.selftrained.hf_transport import download_public_evaluation

                    public_evaluation_source_gate(batch_id, job, manifest_sha, manifest)
                    public_model_dir = Path("/tmp/selftrained-public") / canonical_sha(job["public_export"])
                    public_receipt = download_public_evaluation(
                        job["public_export"],
                        public_model_dir,
                        job["checkpoint"],
                        manifest,
                        job.get("architecture", "moe"),
                    )
                    receipt["public_export"] = public_receipt
                    write_json(output / "public-download-receipt.json", public_receipt)
                command = trainer_command(job, manifest, root, output, batch_id, public_model_dir)
                receipt["command"] = command
                write_json(output / "execution.json", receipt)
                seconds = job.get("max_seconds", resource_spec(stage, job.get("wall_seconds"))["seconds"] - 180)
                with (output / "runner.log").open("w") as stream:
                    process = subprocess.Popen(
                        command, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"}
                    )
                    try:
                        returncode = process.wait(timeout=seconds)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=15)
                        raise RuntimeError(
                            "Bounded runtime expired; saved checkpoints retained; completion is not claimed"
                        ) from None
                receipt["returncode"] = returncode
                if returncode:
                    raise RuntimeError(f"Actual trainer/evaluator exited {returncode}; inspect raw runner.log")
                required = (
                    (
                        "train-receipt.json",
                        "latest.pt",
                        "best.pt",
                        "model.safetensors",
                        "model-config.json",
                        "tokenizer.json",
                        "inference-manifest.json",
                    )
                    if stage in TRAIN_STAGES
                    else ("metrics.json", "outputs.jsonl", "evaluation-receipt.json")
                )
                if any(not (output / name).is_file() for name in required):
                    raise RuntimeError(
                        "Successful process did not produce the actual required trainer/evaluator evidence"
                    )
                if stage in TRAIN_STAGES:
                    training = json.loads((output / "train-receipt.json").read_text())
                    if training.get("steps") != job.get("steps", 300):
                        raise RuntimeError(
                            "Trainer stopped before requested stage steps; saved state retained without completion claim"
                        )
                else:
                    evaluation = json.loads((output / "evaluation-receipt.json").read_text())
                    metrics = json.loads((output / "metrics.json").read_text())
                    if (
                        evaluation.get("status") != "complete"
                        or evaluation.get("completed_count") != evaluation.get("expected_count")
                        or metrics.get("evaluation_complete") is not True
                        or metrics.get("count") != evaluation.get("completed_count")
                    ):
                        raise RuntimeError(
                            "Evaluation interrupted before all selected records; raw prefix and continuation receipt retained"
                        )
                if stage == "freeze" and not (output / "frozen.json").is_file():
                    raise RuntimeError("Validation did not generate the frozen pre-test protocol")
            receipt["status"] = "completed"
        except BaseException as caught:
            error = caught
            receipt.update(status="failed", error_type=type(caught).__name__, error=str(caught))
        finally:
            stop.set()
            thread.join(timeout=5)
            receipt["elapsed_seconds"] = time.monotonic() - started
            receipt["finished_at"] = datetime.now(UTC).isoformat()
            write_json(output / "execution.json", receipt)
            files = [path for path in output.rglob("*") if path.is_file() and not path.is_symlink()]
            if sum(path.stat().st_size for path in files) > MAX_BACKUP_BYTES:
                receipt.update(status="failed", error="Private attempt output exceeded 512 MiB allowance")
                error = error or RuntimeError(receipt["error"])
            receipt["files"] = [
                {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in files
            ]
            write_json(output / "receipt.json", receipt)
            volume.commit()
            volume.reload()
            ledger = json.loads(LEDGER.read_text())
            finish_entry(ledger, run_id, receipt["status"], receipt)
            write_json(LEDGER, ledger)
            volume.commit()
        return receipt

    @app.function(
        serialized=True,
        image=prepare_image,
        cpu=(2, 2),
        memory=(4096, 4096),
        volumes={"/course": volume},
        timeout=300,
        retries=0,
        max_containers=1,
        scaledown_window=2,
    )
    def prepare_remote(*args):
        return execute("prepare", *args)

    @app.function(
        serialized=True,
        image=image,
        gpu="L4",
        cpu=(2, 2),
        memory=(8192, 8192),
        volumes={"/course": volume},
        timeout=900,
        retries=0,
        max_containers=1,
        scaledown_window=2,
    )
    def gpu_remote(stage, *args):
        if phase != "execute" or stage not in GPU_STAGES:
            raise ValueError("GPU execution needs separate exact reservation and execute image phase")
        return execute(stage, *args)

    @app.function(
        serialized=True,
        image=control,
        cpu=(1, 1),
        memory=(2048, 2048),
        volumes={"/course": volume},
        timeout=600,
        retries=0,
        max_containers=1,
        scaledown_window=2,
        secrets=[modal.Secret.from_name("codex_cloud", required_keys=["HF_TOKEN"])],
    )
    def release_remote(*args):
        return execute("release", *args)

    def run_local(
        mode: str,
        run_id: str,
        revision: str,
        manifest: str = "docs/selftrained/manifest.json",
        job: str = "docs/selftrained/jobs/prepare.json",
        batch_id: str = "selftrained-v1",
    ):
        safe_name(run_id)
        safe_name(batch_id)
        output = ROOT / "outputs/selftrained" / run_id
        if mode == "finish":
            write_json(output / "budget-final.json", finish_remote.remote(run_id, "failed-or-cancelled"))
            return
        manifest_value, manifest_sha = committed_json(manifest, revision)
        job_value, job_sha = committed_json(job, revision)
        validate_manifest(manifest_value)
        validate_job(job_value)
        source_hashes = source_gate(revision)
        if mode == "reserve":
            live = read_live_ledger()
            write_json(output / "live-ledger-before.json", live)
            # Avoid a paid reservation container for exhausted budgets or an
            # exact job whose successful evidence already exists.
            reserve_entry(
                json.loads(json.dumps(live["ledger"])),
                run_id,
                batch_id,
                revision,
                manifest_sha,
                job_value,
                job_sha,
                live["live"],
                source_hashes,
            )
            result = reserve_remote.remote(
                run_id,
                batch_id,
                revision,
                manifest_value,
                manifest_sha,
                job_value,
                job_sha,
                live["live"],
                source_hashes,
                live["ledger_sha256"],
            )
            write_json(output / "reservation.json", result)
            print(
                json.dumps(
                    {
                        "stage": job_value["stage"],
                        "reserved_usd": result["entry"]["reserved_usd"],
                        "reserved_total_usd": result["reserved_total_usd"],
                    }
                )
            )
            return
        if mode != "execute":
            raise ValueError("Mode must be reserve, execute or finish")
        args = (run_id, batch_id, revision, manifest_value, manifest_sha, job_value, job_sha)
        if job_value["stage"] == "prepare":
            receipt = prepare_remote.remote(*args)
        elif job_value["stage"] == "release":
            receipt = release_remote.remote(*args)
        else:
            wall_seconds = resource_spec(job_value["stage"], job_value.get("wall_seconds"))["seconds"]
            receipt = gpu_remote.with_options(timeout=wall_seconds).remote(job_value["stage"], *args)
        write_json(output / "receipt.json", receipt)
        total = 0
        for item in receipt.get("files", []):
            path = relative_path(item["path"])
            if path.suffix not in (".json", ".jsonl", ".log") or len(path.parts) != 1:
                continue
            total += item["bytes"]
            if total > MAX_REVIEW_BYTES:
                raise ValueError("Raw review evidence exceeds 64 MiB; full evidence remains on private Volume")
            target = output / "raw" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as stream:
                for block in volume.read_file(f"selftrained/{batch_id}/{job_value['stage']}/{run_id}/{path}"):
                    stream.write(block)
            if sha256(target) != item["sha256"] or target.stat().st_size != item["bytes"]:
                raise RuntimeError("Downloaded raw receipt differs from actual committed Volume bytes")
        if receipt["status"] != "completed":
            raise RuntimeError("Actual bounded attempt failed; raw evidence and full reservation retained")

    return app, run_local


def main(
    mode: str,
    run_id: str,
    revision: str,
    manifest: str = "docs/selftrained/manifest.json",
    job: str = "docs/selftrained/jobs/prepare.json",
    batch_id: str = "selftrained-v1",
):
    return _run_local(mode, run_id, revision, manifest, job, batch_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ledger", "check"))
    parser.add_argument("--revision")
    parser.add_argument("--manifest", default="docs/selftrained/manifest.json")
    parser.add_argument("--job", default="docs/selftrained/jobs/prepare.json")
    parser.add_argument("--output", default="outputs/selftrained/live-ledger.json")
    options = parser.parse_args()
    if options.mode == "ledger":
        result = read_live_ledger()
        write_json(ROOT / relative_path(options.output), result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("observed_at", "ledger_sha256", "reserved_total_usd", "remaining_reserved_budget_usd")
                }
            )
        )
    else:
        manifest_value, manifest_sha = committed_json(options.manifest, options.revision)
        job_value, job_sha = committed_json(options.job, options.revision)
        validate_manifest(manifest_value)
        validate_job(job_value)
        source_gate(options.revision)
        print(
            json.dumps(
                {
                    "stage": job_value["stage"],
                    "manifest_sha256": manifest_sha,
                    "job_sha256": job_sha,
                    "source_ready": True,
                }
            )
        )
elif os.environ.get("SELFTRAINED_MODAL_PHASE"):
    app, _run_local = register_modal()
    main = app.local_entrypoint()(main)
