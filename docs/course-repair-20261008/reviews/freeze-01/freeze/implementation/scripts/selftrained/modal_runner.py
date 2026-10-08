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
sys.path.insert(0, str(ROOT))
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
GROSS_QUOTA = COURSE / "gross-quota.json"
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
    quota_sha = ledger.get("gross_quota_policy_sha256")
    if quota_sha is not None and not re.fullmatch(r"[a-f0-9]{64}", quota_sha):
        raise ValueError("Invalid gross quota policy marker in legacy audit ledger")
    if total > TOTAL_CAP_USD and quota_sha is None:
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
    for key in ("tool_loss_weight", "numeric_run_loss_weight", "native_voice_loss_weight"):
        if key in job:
            value = job[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 1 or stage != "joint":
                raise ValueError(f"{key} must be a finite number >= 1 and is available only for joint training")
    if job.get("native_voice_loss_weight", 1) != 1:
        if (
            job["native_voice_loss_weight"] != 4
            or job.get("tool_loss_weight", 1) != 4
            or job.get("numeric_run_loss_weight", 1) != 1
            or job.get("steps", 300) > 4000
            or job.get("batch_size", 16) != 16
            or job.get("context", 512) != 512
            or learning_rate != Decimal("0.0002")
            or sampling != "task-family"
            or freeze is not True
            or spec["seconds"] != 900
            or seconds > 720
            or job.get("eval_every", 100) != 1000
            or job.get("save_every", 100) != 1000
        ):
            raise ValueError(
                "native voice candidate is fixedtool4/numeric1/native4, <=4000 steps, batch16/context512/LR0.0002, frozen task-family, eval/save1000, wall900/max720"
            )
        descriptor = job.get("init_checkpoint") or job.get("resume") or {}
        if descriptor.get("stage") != "joint" or (job.get("init_checkpoint") and descriptor.get("path") != "best.pt"):
            raise ValueError(
                "native voice must init own completed selected joint best, or exact same-objective joint resume"
            )
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
        from scripts.selftrained.hf_transport import validate_batch_release, validate_repository_card_release

        release = job.get("release", {})
        if release.get("mode") == "repository-card":
            validate_repository_card_release(release)
            if set(job) != {"schema_version", "stage", "release", "gross_quota_policy"}:
                raise ValueError("Repository card needs only its release descriptor and explicit gross policy")
        else:
            if "mode" in release:
                raise ValueError("Unknown release mode")
            for export in validate_batch_release(release):
                validate_descriptor(export["source"], checkpoint=True)
    if "gross_quota_policy" in job:
        policy = job["gross_quota_policy"]
        if not isinstance(policy, dict) or set(policy) != {"path", "sha256"}:
            raise ValueError("Gross quota policy needs an exact committed path/SHA descriptor")
        path = relative_path(policy["path"])
        if path.parts[:2] != ("docs", "selftrained") or not re.fullmatch(r"[a-f0-9]{64}", policy["sha256"]):
            raise ValueError("Gross quota policy must be committed under docs/selftrained")
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


def job_quota_policy(job, revision):
    descriptor = job.get("gross_quota_policy")
    if descriptor is None:
        return None, None
    from scripts.selftrained.finance import validate_policy

    policy, policy_sha = committed_json(descriptor["path"], revision)
    if policy_sha != descriptor["sha256"]:
        raise ValueError("Gross quota policy differs from committed job descriptor")
    validate_policy(policy, policy_sha)
    return policy, policy_sha


def committed_repository_card(job, revision, manifest_sha):
    """Read the one card from the actual dispatch Git blob, before remote work."""
    from scripts.selftrained.hf_transport import approved_repository_card, validate_repository_card_release

    release = job.get("release", {})
    if release.get("mode") != "repository-card":
        return None
    item = validate_repository_card_release(release)
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Repository card requires the actual full dispatch Git SHA")
    source = ROOT / item["path"]
    if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Repository card source is absent or outside the repository")
    if source.stat().st_size != item["bytes"]:
        raise ValueError("Repository card source exceeds or differs from reviewed bytes")
    size = subprocess.run(
        ["git", "cat-file", "-s", f"{revision}:{item['path']}"], cwd=ROOT, capture_output=True, check=True, text=True
    ).stdout.strip()
    if size != str(item["bytes"]):
        raise ValueError("Committed card blob size differs from bounded descriptor")
    payload = subprocess.run(
        ["git", "show", f"{revision}:{item['path']}"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    if source.read_bytes() != payload:
        raise ValueError("Repository card source differs from the exact dispatch Git blob")
    approved_repository_card(payload, release, manifest_sha)
    return payload


def repository_card_source_gate(payload, release, revision, manifest_sha, source_hashes):
    from scripts.selftrained.hf_transport import approved_repository_card, repository_card_public_gate

    item = approved_repository_card(payload, release, manifest_sha)
    if not re.fullmatch(r"[a-f0-9]{40}", revision) or source_hashes.get(item["path"]) != item["sha256"]:
        raise ValueError("Repository card bytes are not bound to the actual dispatch source SHA")
    return repository_card_public_gate(payload, release)


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


def read_live_ledger(quota_policy=None, quota_policy_sha=None):
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
    quota_state, quota_state_sha = None, None
    if quota_policy is not None:
        from scripts.selftrained.finance import official_snapshot

        try:
            quota_chunks, quota_bytes = [], 0
            for chunk in shared.read_file("gross-quota.json"):
                quota_bytes += len(chunk)
                if quota_bytes > MAX_LEDGER_BYTES:
                    raise ValueError("Gross quota sidecar exceeds bounded review size")
                quota_chunks.append(chunk)
            quota_raw = b"".join(quota_chunks)
        except (FileNotFoundError, modal.exception.NotFoundError):
            quota_raw = None
        if quota_raw is not None:
            if len(quota_raw) > MAX_LEDGER_BYTES:
                raise ValueError("Gross quota sidecar exceeds bounded review size")
            quota_state = json.loads(quota_raw)
            quota_state_sha = hashlib.sha256(quota_raw).hexdigest()
        if bool(ledger.get("gross_quota_policy_sha256")) != bool(quota_state is not None):
            raise RuntimeError("Gross quota sidecar and legacy marker disagree; reconcile without reset")
        live["gross_quota_snapshot"] = official_snapshot(modal.Workspace.from_context(), quota_policy, quota_policy_sha)
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
        **(
            {
                "gross_quota_state": quota_state,
                "gross_quota_state_sha256": quota_state_sha,
                "legacy_totals_audit_only": True,
                "gross_quota_authorized_ceiling_usd": str(
                    Decimal(quota_policy["baseline"]["gross_usd"]) + Decimal(quota_policy["additional_quota_usd"])
                ),
            }
            if quota_policy is not None
            else {}
        ),
    }


def reserve_entry(
    ledger,
    run_id,
    batch_id,
    revision,
    manifest_sha,
    job,
    job_sha,
    live,
    source_hashes,
    quota_policy=None,
    quota_policy_sha=None,
    quota_state=None,
):
    prior = ledger_total(ledger)
    if any(entry.get("run_id") == run_id for entry in ledger["reservations"]):
        raise ValueError("Each attempt needs a fresh run ID; prior reservations are never reused")
    numerical_job = {
        key: value for key, value in job.items() if key not in ("wall_seconds", "max_seconds", "gross_quota_policy")
    }
    logical_sha = canonical_sha({"batch_id": batch_id, "manifest_sha256": manifest_sha, "job": numerical_job})
    if any(
        entry.get("logical_job_sha256") == logical_sha and entry.get("status") == "completed"
        for entry in ledger["reservations"]
    ):
        raise ValueError("This exact job already completed; do not rerun successful numbers")
    guard = reservation_guard(job["stage"], live["rates"], job.get("wall_seconds"))
    amount = Decimal(guard["reserved_usd"])
    quota_next, quota_guard = None, None
    descriptor = job.get("gross_quota_policy")
    marker = ledger.get("gross_quota_policy_sha256")
    if descriptor is not None or marker is not None or quota_policy is not None:
        from scripts.selftrained.finance import check_quota

        if descriptor is None or quota_policy is None or descriptor["sha256"] != quota_policy_sha:
            raise ValueError("New round requires the same explicit gross quota policy on client and server")
        if marker is not None and marker != quota_policy_sha:
            raise ValueError("Gross quota epoch cannot replace an existing policy")
        if bool(marker) != bool(quota_state is not None):
            raise ValueError("Gross quota sidecar and existing marker must agree")
        quota_next, quota_guard = check_quota(
            quota_policy, quota_policy_sha, quota_state, live["gross_quota_snapshot"], ledger, amount, run_id
        )
    elif prior + amount > TOTAL_CAP_USD:
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
    if quota_next is not None:
        entry.update(
            gross_quota_policy_sha256=quota_policy_sha,
            gross_quota_guard=quota_guard,
            scope="Official gross increment plus temporary current-round bounds; legacy reserved totals are audit only",
        )
        ledger["gross_quota_policy_sha256"] = quota_policy_sha
    ledger["reservations"].append(entry)
    ledger.update(budget_usd=str(TOTAL_CAP_USD), reserved_total_usd=str(prior + amount))
    return (entry, quota_next) if quota_next is not None else entry


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


def native_objective_metadata(native=4, numeric=1):
    """Pure JSON canonical policy for control preflight; must match the trainer exactly."""
    policy = {
        "version": "selftrained-language-objective-v1",
        "training_only": True,
        "legacy_default_branch": "reuse_original_language_loss_scalar_and_graph",
        "tool_rows": ["tool_call", "tool_reply:supervision.replay_kind=actual_executor"],
        "excluded_rows": ["evaluation_only", "supervision.protocol_fixture"],
        "row_scope": "all_existing_supervised_assistant_targets_in_eligible_row",
        "numeric_targets": "ASCII_digits_union_immediate_supervised_nondigit_boundary_or_EOS_union_minus_if_next_supervised_digit",
        "alignment": "RecordEncoder_already_shifted_labels; ignored_gap_breaks_adjacency",
        "combination": "tool_row_multiplier_times_numeric_multiplier; ignored_labels_zero",
        "normalization": "sum(weight_times_token_CE)/sum(weight)",
        "weighted_reduction_dtype": "float32",
        "validation_and_selection": "original_unweighted_objective; validation_loss",
    }
    metadata = {"tool_loss_weight": 4, "numeric_run_loss_weight": numeric, "language_objective_policy": policy}
    if native != 1:
        policy.update(
            version="selftrained-language-objective-v2",
            native_one_branch="exact_existing_v1_scalar_graph_metadata_and_logs",
            native_voice_rows={
                "split": "train",
                "tasks": ["voice_qa", "voice_topic_continuation"],
                "augmentation": "key_absent",
                "excluded": ["evaluation_only", "supervision.protocol_fixture"],
            },
            native_row_scope="all_existing_supervised_assistant_targets_including_public_history_confirmations",
            combination="tool_row_multiplier_times_numeric_multiplier_times_native_voice_row_multiplier; ignored_labels_zero",
        )
        metadata["native_voice_loss_weight"] = native
    return metadata


def native_target_source_options_gate(training, source_job, target_job, manifest=None, resume=False):
    """Compare inherited target options before reservation; no torch or private checkpoint load."""
    if training.get("config", {}).get("architecture") != source_job.get("architecture", "moe") or training.get(
        "config", {}
    ).get("max_length") != source_job.get("context", 512):
        raise ValueError("native source config/options differ before reserve")
    for name, default in (
        ("architecture", "moe"),
        ("batch_size", 16),
        ("seed", 20261006),
        ("learning_rate", 0.001),
        ("weight_decay", 0.01),
        ("perception_weight", 1.0),
        ("router_weight", 0.01),
        ("freeze_perception_backbones", False),
        ("sampling_mode", "bucket"),
        ("context", 512),
    ):
        actual, wanted = source_job.get(name, default), target_job.get(name, default)
        if name in ("learning_rate", "weight_decay", "perception_weight", "router_weight"):
            actual, wanted = Decimal(str(actual)), Decimal(str(wanted))
        recorded = training.get(name, source_job.get(name, default))
        if name in ("learning_rate", "weight_decay", "perception_weight", "router_weight"):
            recorded = Decimal(str(recorded))
        if actual != wanted or recorded != actual:
            raise ValueError(f"native target/source execution/receipt options differ before reserve: {name}")
    if resume:
        for name in ("eval_every", "save_every"):
            if target_job.get(name, 100) != source_job.get(name, 100):
                raise ValueError(f"native exact resume changes {name} before reserve")
        if target_job.get("steps", 300) < training["steps"]:
            raise ValueError("native resume target steps precede source saved steps")
    if manifest is not None:
        records = {Path(item["path"]).name: item["sha256"] for item in manifest["records"]}
        assets = {item["path"]: item["sha256"] for item in manifest.get("assets", [])}
        if (
            training["data_sha256"] != records
            or any(assets.get(name) != digest for name, digest in training["asset_sha256"].items())
            or any(
                training["config"].get(key) != value
                for key, value in manifest["model_config"].items()
                if key not in ("vocab_size", "architecture")
            )
        ):
            raise ValueError("native target frozen data/assets/config differ before reserve")


def native_joint_source_gate(path, architecture, descriptor=None, manifest_sha=None, target_job=None, manifest=None):
    """Native fresh init binds actual same-directory Volume receipts; no checkpoint deserialization."""
    from scripts.selftrained.hf_transport import verify_file

    path = Path(path)
    if path.name != "best.pt" or path.is_symlink() or not path.is_file():
        raise ValueError("native fresh source must be actual selected best.pt")
    root = path.parent
    execution = json.loads((root / "execution.json").read_text())
    receipt = json.loads((root / "receipt.json").read_text())
    if (
        execution.get("status") != "completed"
        or execution.get("returncode") != 0
        or execution.get("stage") != "joint"
        or any(
            receipt.get(key) != execution.get(key)
            for key in ("status", "revision", "stage", "run_id", "manifest_sha256", "job")
        )
        or (manifest_sha is not None and execution.get("manifest_sha256") != manifest_sha)
    ):
        raise ValueError("native source needs its actual completed matching execution/receipt")
    job = execution["job"]
    if (
        job.get("stage") != "joint"
        or job.get("architecture", "moe") != architecture
        or job.get("tool_loss_weight", 1) != 4
        or job.get("numeric_run_loss_weight", 1) != 4
        or job.get("native_voice_loss_weight", 1) != 1
        or Decimal(str(job.get("learning_rate", "0.001"))) != Decimal("0.0002")
        or job.get("batch_size", 16) != 16
        or job.get("context", 512) != 512
        or job.get("freeze_perception_backbones") is not True
        or job.get("sampling_mode") != "task-family"
    ):
        raise ValueError(
            "native source must be the own completed reviewed4/4 joint with unchanged LR/batch/context/freeze/family"
        )
    recorded = receipt.get("files", [])
    entries = {item["path"]: item for item in recorded}
    if len(entries) != len(recorded):
        raise ValueError("native source receipt repeats a recorded raw path")
    required = (
        "best.pt",
        "execution.json",
        "train-receipt.json",
        "inference-manifest.json",
        "model-config.json",
        "tokenizer.json",
        "model.safetensors",
    )
    if any(name not in entries for name in required):
        raise ValueError("native source is missing its recorded selected checkpoint/metadata")
    paths = {name: verify_file(root, entries[name]) for name in required}
    selected_sha = entries["best.pt"]["sha256"]
    if descriptor is not None and (
        descriptor.get("sha256") != selected_sha
        or descriptor.get("stage") != "joint"
        or descriptor.get("path") != "best.pt"
        or descriptor.get("run_id") != execution.get("run_id")
    ):
        raise ValueError("native source differs from its exact selected descriptor")
    training = json.loads(paths["train-receipt.json"].read_text())
    inference = json.loads(paths["inference-manifest.json"].read_text())
    config = json.loads(paths["model-config.json"].read_text())
    tokenizer = json.loads(paths["tokenizer.json"].read_text())
    tokenizer_sha = hashlib.sha256(json.dumps(tokenizer, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if (
        training.get("stage") != "joint"
        or training.get("architecture") != architecture
        or training.get("completed_requested_steps") is not True
        or training.get("interrupted") is not False
        or training.get("selected_checkpoint_available") is not True
        or training.get("inference_exported") is not True
        or training.get("test_used_for_selection") is not False
        or type(training.get("steps")) is not int
        or training["steps"] != job.get("steps", 300)
        or type(inference.get("selected_step")) is not int
        or not 0 < inference["selected_step"] <= training["steps"]
        or inference.get("selected_checkpoint_sha256") != selected_sha
        or inference.get("stage") != "joint"
        or inference.get("selection") != "validation_loss"
        or training.get("origin", {}).get("kind") != "all-neural-weights-random"
        or training.get("origin") != inference.get("origin")
        or config.get("architecture") != architecture
        or training.get("config") != config
        or training.get("tokenizer_sha256") != tokenizer_sha
        or inference.get("files")
        != {name: entries[name]["sha256"] for name in ("model.safetensors", "model-config.json", "tokenizer.json")}
        or any(
            training.get(key) != inference.get(key)
            for key in (
                "schema",
                "data_sha256",
                "asset_sha256",
                "tokenizer_sha256",
                "freeze_perception_backbones",
                "sampling_mode",
                "tool_loss_weight",
                "numeric_run_loss_weight",
                "language_objective_policy",
            )
        )
        or training.get("tool_loss_weight") != 4
        or training.get("numeric_run_loss_weight") != 4
        or training.get("native_voice_loss_weight", 1) != 1
        or inference.get("native_voice_loss_weight", 1) != 1
        or training.get("language_objective_policy") != native_objective_metadata(1, 4)["language_objective_policy"]
        or training.get("seed") != job.get("seed", 20261006)
    ):
        raise ValueError("native source selected export and completed training identity/options differ")
    native_target_source_options_gate(training, job, job if target_job is None else target_job, manifest)
    return {
        "training": training,
        "inference": inference,
        "job": job,
        "binding": {
            "source_checkpoint_sha256": selected_sha,
            "source_run_id": execution["run_id"],
            "source_revision": execution["revision"],
            "source_manifest_sha256": execution["manifest_sha256"],
            "source_execution_sha256": sha256(root / "execution.json"),
            "source_receipt_sha256": sha256(root / "receipt.json"),
            "source_train_receipt_sha256": entries["train-receipt.json"]["sha256"],
            "source_inference_manifest_sha256": entries["inference-manifest.json"]["sha256"],
            "source_selected_step": inference["selected_step"],
            "source_completed_steps": training["steps"],
        },
    }


def native_resume_source_gate(path, target_job, descriptor=None, manifest_sha=None, manifest=None):
    """A pinned incomplete V2 branch may resume; V1 or changed options/policy may not reserve."""
    from scripts.selftrained.hf_transport import verify_file

    path = Path(path)
    if path.name not in ("latest.pt", "best.pt") or path.is_symlink() or not path.is_file():
        raise ValueError("native resume needs its actual full saved checkpoint")
    root = path.parent
    execution = json.loads((root / "execution.json").read_text())
    receipt = json.loads((root / "receipt.json").read_text())
    if (
        execution.get("stage") != "joint"
        or execution.get("status") not in ("failed", "completed")
        or any(
            receipt.get(key) != execution.get(key)
            for key in ("status", "revision", "stage", "run_id", "manifest_sha256", "job")
        )
        or (manifest_sha is not None and execution.get("manifest_sha256") != manifest_sha)
    ):
        raise ValueError("native resume needs matching actual source execution/receipt")
    recorded = receipt.get("files", [])
    entries = {item["path"]: item for item in recorded}
    if len(entries) != len(recorded) or any(
        name not in entries for name in (path.name, "execution.json", "train-receipt.json")
    ):
        raise ValueError("native resume source lacks its actual recorded checkpoint/receipt")
    for name in (path.name, "execution.json", "train-receipt.json"):
        verify_file(root, entries[name])
    if descriptor is not None and (
        descriptor.get("sha256") != entries[path.name]["sha256"]
        or descriptor.get("run_id") != execution.get("run_id")
        or descriptor.get("stage") != "joint"
        or descriptor.get("path") != path.name
    ):
        raise ValueError("native resume descriptor differs from actual saved source SHA")
    training = json.loads((root / "train-receipt.json").read_text())
    job = execution["job"]
    expected = native_objective_metadata()
    if (
        training.get("schema") != "selftrained-random-v1"
        or training.get("stage") != "joint"
        or training.get("architecture") != target_job.get("architecture", "moe")
        or training.get("config", {}).get("architecture") != target_job.get("architecture", "moe")
        or training.get("origin", {}).get("kind") != "all-neural-weights-random"
        or training.get("test_used_for_selection") is not False
        or type(training.get("steps")) is not int
        or not 0 < training["steps"] <= job.get("steps", 300)
        or any(training.get(key) != value for key, value in expected.items())
        or any(job.get(key, 1) != value for key, value in expected.items() if key != "language_objective_policy")
        or any(target_job.get(key, 1) != value for key, value in expected.items() if key != "language_objective_policy")
    ):
        raise ValueError("native exact resume source/target objective/full policy differs before reserve")
    if path.name == "best.pt":
        if "inference-manifest.json" not in entries:
            raise ValueError("native selected resume lacks exact selected provenance")
        inference = json.loads(verify_file(root, entries["inference-manifest.json"]).read_text())
        if inference.get("selected_checkpoint_sha256") != entries[path.name]["sha256"] or any(
            inference.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("native selected resume objective/SHA differs")
    native_target_source_options_gate(training, job, target_job, manifest, resume=True)
    return {
        "source_checkpoint_sha256": entries[path.name]["sha256"],
        "source_steps": training["steps"],
        "objective": expected,
    }


def native_resume_preflight_required(job, source_execution, path):
    """Source V2 must still be guarded when target omits or resets its native coefficient."""
    if (
        job.get("native_voice_loss_weight", 1) != 1
        or source_execution.get("job", {}).get("native_voice_loss_weight", 1) != 1
    ):
        return True
    training_path = Path(path).parent / "train-receipt.json"
    if training_path.is_file():
        try:
            training = json.loads(training_path.read_text())
        except (OSError, ValueError):
            return False  # Keep unrelated V1 legacy resume behavior unchanged.
        return training.get("language_objective_policy", {}).get("version") == "selftrained-language-objective-v2"
    return False


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
        for flag in ("tool_loss_weight", "numeric_run_loss_weight", "native_voice_loss_weight"):
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
        run_id,
        batch_id,
        revision,
        manifest,
        manifest_sha,
        job,
        job_sha,
        live,
        source_hashes,
        previous_sha,
        quota_policy=None,
        quota_policy_sha=None,
        quota_previous_sha=None,
        repository_card_bytes=None,
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
                if field == "init_checkpoint" and job.get("native_voice_loss_weight", 1) != 1:
                    native_joint_source_gate(
                        path,
                        job.get("architecture", "moe"),
                        job[field],
                        manifest_sha,
                        target_job=job,
                        manifest=manifest,
                    )
                if field == "resume" and native_resume_preflight_required(job, source_execution, path):
                    native_resume_source_gate(path, job, job[field], manifest_sha, manifest)
                if field == "protocol":
                    evaluation_public_lineage_gate(source_execution, job)
                    test_protocol_gate(path, job, manifest, source_hashes, resume_path)
        if job["stage"] == "release":
            if job["release"].get("mode") == "repository-card":
                gate = repository_card_source_gate(
                    repository_card_bytes, job["release"], revision, manifest_sha, source_hashes
                )
                if gate["no_op"]:
                    raise ValueError(
                        "Repository card already matches the pinned parent; use free preflight, no reservation"
                    )
            else:
                release_sources_gate(batch_id, job["release"], manifest_sha, manifest)
        ledger = json.loads(LEDGER.read_text())
        quota_state = None
        if quota_policy is not None:
            from scripts.selftrained.finance import official_snapshot

            current_quota_sha = sha256(GROSS_QUOTA) if GROSS_QUOTA.is_file() else None
            if current_quota_sha != quota_previous_sha:
                raise RuntimeError("Gross quota sidecar changed since the client probe")
            quota_state = json.loads(GROSS_QUOTA.read_text()) if GROSS_QUOTA.is_file() else None
            live = {
                **live,
                "gross_quota_snapshot": official_snapshot(
                    modal.Workspace.from_context(), quota_policy, quota_policy_sha
                ),
            }
        result = reserve_entry(
            ledger,
            run_id,
            batch_id,
            revision,
            manifest_sha,
            job,
            job_sha,
            live,
            source_hashes,
            quota_policy,
            quota_policy_sha,
            quota_state,
        )
        entry, quota_next = result if quota_policy is not None else (result, None)
        if quota_next is not None:
            write_json(GROSS_QUOTA, quota_next)
        write_json(LEDGER, ledger)
        volume.commit()
        return {
            "entry": entry,
            "budget_usd": ledger["budget_usd"],
            "reserved_total_usd": ledger["reserved_total_usd"],
            **({"gross_quota_guard": entry["gross_quota_guard"]} if "gross_quota_guard" in entry else {}),
        }

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

    def execute(stage, run_id, batch_id, revision, manifest, manifest_sha, job, job_sha, repository_card_bytes=None):
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
        if stage == "release" and job["release"].get("mode") == "repository-card":
            repository_card_source_gate(
                repository_card_bytes, job["release"], revision, manifest_sha, entry["source_sha256"]
            )
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
                from scripts.selftrained.hf_transport import publish_batch_inference, publish_repository_card

                if job["release"].get("mode") == "repository-card":
                    result = publish_repository_card(
                        repository_card_bytes, job["release"], manifest_sha, os.environ["HF_TOKEN"]
                    )
                else:
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
        repository_card_bytes = committed_repository_card(job_value, revision, manifest_sha)
        if repository_card_bytes is not None:
            source_hashes[job_value["release"]["file"]["path"]] = hashlib.sha256(repository_card_bytes).hexdigest()
        quota_policy, quota_policy_sha = job_quota_policy(job_value, revision)
        if mode == "reserve":
            live = read_live_ledger(quota_policy, quota_policy_sha)
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
                quota_policy,
                quota_policy_sha,
                live.get("gross_quota_state"),
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
                quota_policy,
                quota_policy_sha,
                live.get("gross_quota_state_sha256"),
                **({"repository_card_bytes": repository_card_bytes} if repository_card_bytes is not None else {}),
            )
            write_json(output / "reservation.json", result)
            print(
                json.dumps(
                    {
                        "stage": job_value["stage"],
                        "reserved_usd": result["entry"]["reserved_usd"],
                        "reserved_total_usd": result["reserved_total_usd"],
                        **(
                            {"legacy_totals_audit_only": True, "gross_quota_guard": result["gross_quota_guard"]}
                            if "gross_quota_guard" in result
                            else {}
                        ),
                    }
                )
            )
            return
        if mode != "execute":
            raise ValueError("Mode must be reserve, execute or finish")
        args = (run_id, batch_id, revision, manifest_value, manifest_sha, job_value, job_sha)
        if repository_card_bytes is not None:
            args += (repository_card_bytes,)
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
        committed_repository_card(job_value, options.revision, manifest_sha)
        job_quota_policy(job_value, options.revision)
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
