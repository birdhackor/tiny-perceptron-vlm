"""Bounded, versioned natural-image/OCR/speech assistant experiments.

The reserve invocation uses a small control image. All remote entrypoints are
defined in every import context. Only the local execute invocation constructs
the CUDA image recipe, after the separate reservation invocation has succeeded.
It shares the original course volume and ledger; a new batch never resets spend.
This module does not read or export the Modal credential or the HF token value.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME_ROOT = Path("/course")
NATURAL_ROOT = VOLUME_ROOT / "natural-v3"
LEDGER_PATH = VOLUME_ROOT / "budget.json"
TOTAL_CAP_USD = Decimal("40.00")
PRIOR_RESERVED_FLOOR_USD = Decimal("9.84")
MAX_JOB_USD = Decimal("2.00")
STAGES = ("prepare", "baseline", "train", "validation", "evaluate", "external_prepare", "external_ocr", "release")
SELECTION_STAGES = ("evaluate", "external_ocr")
EXTERNAL_STAGES = ("external_prepare", "external_ocr")
SPEC = {
    "prepare": {"cpu": 2, "memory_gib": 8, "seconds": 1800, "gpu": False},
    "baseline": {"cpu": 4, "memory_gib": 32, "seconds": 3600, "gpu": True},
    "train": {"cpu": 4, "memory_gib": 32, "seconds": 3600, "gpu": True},
    "validation": {"cpu": 4, "memory_gib": 32, "seconds": 3600, "gpu": True},
    "evaluate": {"cpu": 4, "memory_gib": 32, "seconds": 3600, "gpu": True},
    "external_prepare": {"cpu": 2, "memory_gib": 8, "seconds": 1800, "gpu": False},
    "external_ocr": {"cpu": 4, "memory_gib": 32, "seconds": 3600, "gpu": True},
    "release": {"cpu": 1, "memory_gib": 4, "seconds": 1800, "gpu": False},
}
# These allowances are reservations, not measured invoice amounts. Network and
# storage are capped separately below; unchanged retained caches remain subject
# to workspace billing rather than a claim of perpetual free storage.
AUXILIARY_SECONDS = 3600
BUILD_AND_RETAINED_STORAGE_ALLOWANCE_USD = Decimal("0.20")
MAX_EGRESS_GIB = Decimal("8")
MAX_PRIVATE_BACKUP_BYTES = 256 * 1024 * 1024
MAX_REVIEW_DOWNLOAD_BYTES = 64 * 1024 * 1024
ASR_VARIANTS = {
    "small": ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d"),
    "turbo": ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
}

PHASE = os.environ.get("NATURAL_MODAL_PHASE", "control")
MANIFEST_RELATIVE = os.environ.get("NATURAL_MANIFEST", "docs/natural-assistant/manifest.json")
EXTERNAL_METADATA_RELATIVE = "docs/natural-assistant/external-ocr.json"
app = modal.App("tiny-perceptron-natural-assistant-v3")
volume = modal.Volume.from_name("tiny-perceptron-course", create_if_missing=False)
hf_secret = modal.Secret.from_name(os.environ.get("HF_MODAL_SECRET") or "codex_cloud", required_keys=["HF_TOKEN"])
control_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface-hub==0.36.2")


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value) or ".." in value:
        raise ValueError("Expected a plain versioned name without path components")
    return value


def safe_relative(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Expected a nonempty relative path without parent traversal")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_names(value):
    if not value:
        return []
    names = value.split(",")
    if (
        len(names) > 2
        or len(set(names)) != len(names)
        or any(not re.fullmatch(r"step-[0-9]{6}", name) or not 1 <= int(name[5:]) <= 3000 for name in names)
    ):
        raise ValueError("Need at most two unique step-NNNNNN checkpoint names")
    return names


def runtime_options(
    stage, steps, seed, max_seconds, max_pixels, learning_rate, asr_variant, checkpoints, checkpoint, compare
):
    if (
        type(steps) is not int
        or not 1 <= steps <= 3000
        or type(max_pixels) is not int
        or not 1 <= max_pixels <= 1048576
    ):
        raise ValueError("Requested steps or visual budget exceed the bounded runner")
    limit = min(3300, SPEC[stage]["seconds"] - 180)
    if type(max_seconds) is not int or not 1 <= max_seconds <= limit:
        raise ValueError("Runner timeout must leave at least 180 seconds for checkpoint/cleanup")
    if Decimal(str(learning_rate)) not in (Decimal("0.0001"), Decimal("0.00003")):
        raise ValueError("Learning rate must be exactly 1e-4 or 3e-5")
    if asr_variant not in ASR_VARIANTS:
        raise ValueError("ASR must use the fixed small or turbo model pin")
    saved = []
    if checkpoints:
        fields = checkpoints.split(",")
        if any(not re.fullmatch(r"[0-9]+", field) for field in fields):
            raise ValueError("Checkpoint updates must be comma-separated integers")
        saved = [int(field) for field in fields]
        if stage != "train" or len(saved) > 2 or saved != sorted(set(saved)) or any(not 1 <= n <= steps for n in saved):
            raise ValueError("At most two increasing checkpoint updates within this train run are allowed")
    selected = checkpoint_names(checkpoint)
    if len(selected) > 1:
        raise ValueError("Final inference must identify only one adapter checkpoint")
    compared = checkpoint_names(compare)
    if compared and (stage != "validation" or selected):
        raise ValueError("Checkpoint comparison is validation-only and cannot also select one checkpoint")
    return {
        "steps": steps,
        "seed": seed,
        "max_seconds": max_seconds,
        "max_pixels": max_pixels,
        "learning_rate": float(learning_rate),
        "asr_variant": asr_variant,
        "asr_model": ASR_VARIANTS[asr_variant][0],
        "asr_revision": ASR_VARIANTS[asr_variant][1],
        "checkpoint_steps": saved,
        "adapter_checkpoint": checkpoint,
        "adapter_checkpoints": compared,
    }


def runtime_contract(options):
    return json.dumps(options, sort_keys=True, separators=(",", ":"))


def verify_runtime_contract(stage, options):
    settings = runtime_options(
        stage,
        options["steps"],
        options["seed"],
        options["max_seconds"],
        options["max_pixels"],
        options["learning_rate"],
        options["asr_variant"],
        ",".join(map(str, options["checkpoint_steps"])),
        options["adapter_checkpoint"],
        ",".join(options["adapter_checkpoints"]),
    )
    digest = hashlib.sha256(runtime_contract(settings).encode()).hexdigest()
    if (
        settings != options["runtime_options"]
        or any(options.get(key) != value for key, value in settings.items())
        or digest != options["runtime_options_sha256"]
    ):
        raise ValueError("Actual runtime arguments differ from their canonical reserved configuration")
    return digest


def adapter_descriptor(batch_id, run_id, checkpoint="", manifest_sha=None):
    safe_name(run_id)
    names = checkpoint_names(checkpoint)
    if len(names) > 1:
        raise ValueError("One descriptor identifies exactly one checkpoint")
    directory = NATURAL_ROOT / batch_id / "train" / run_id
    directory = directory / "checkpoints" / checkpoint if checkpoint else directory / "adapter"
    weight = directory / "adapter_model.safetensors"
    if not weight.is_file():
        raise ValueError("Requested adapter is absent in the same explicit batch")
    if checkpoint:
        metadata = json.loads((directory / "checkpoint.json").read_text())
        training = json.loads((directory / "training.json").read_text())
        completed = int(checkpoint[5:])
        if metadata.get("completed_steps") != completed or training.get("completed_steps") != completed:
            raise ValueError("Checkpoint name and actual saved training update differ")
        if manifest_sha and any(item.get("manifest_sha256") != manifest_sha for item in (metadata, training)):
            raise ValueError("Archived checkpoint belongs to a different manifest")
        files = metadata.get("files", [])
        if not {"adapter_model.safetensors", "adapter_config.json", "training.json", "training_state.pt"} <= {
            item.get("path") for item in files
        }:
            raise ValueError("Archived checkpoint must contain complete adapter, metadata and resume state")
        for item in files:
            relative = safe_relative(item["path"])
            if len(relative.parts) != 1:
                raise ValueError("Checkpoint metadata may only name files in its own directory")
            file = directory / relative
            if not file.is_file() or file.stat().st_size != item["bytes"] or sha256(file) != item["sha256"]:
                raise ValueError("Archived checkpoint differs from the exact saved update bytes")
    return {
        "variant": f"adapter-{checkpoint}" if checkpoint else "adapter",
        "checkpoint": checkpoint,
        "path": str(directory),
        "adapter_sha256": sha256(weight),
        "adapter_run_id": run_id,
    }


def snapshot():
    # These calls run in the authenticated Actions client, before a GPU call.
    # No workspace identity, key, credit balance or token enters public evidence.
    billing = modal.Workspace.from_context().billing
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "rates": {key: str(value) for key, value in billing.rates().items()},
        "scope": "Live workspace unit rates; reservation is not an invoice or project billing delta",
    }


def reservation_guard(stage, rates):
    if stage not in STAGES:
        raise ValueError("Unknown bounded stage")
    required = ("cpu_hour_cost", "mem_gib_hour_cost", "egress_gib_cost")
    if SPEC[stage]["gpu"]:
        required += ("gpu_hour_cost_l4",)
    if any(key not in rates for key in required):
        raise ValueError("Live billing API lacks previously verified hourly/GiB rate fields")
    values = {key: Decimal(rates[key]) for key in required}
    if any(not number.is_finite() or number < 0 for number in values.values()):
        raise ValueError("Live prices must be finite and nonnegative")
    spec = SPEC[stage]
    hourly = spec["cpu"] * values["cpu_hour_cost"] + spec["memory_gib"] * values["mem_gib_hour_cost"]
    if spec["gpu"]:
        hourly += values["gpu_hour_cost_l4"]
    compute = Decimal(spec["seconds"] + 2) * hourly / Decimal(3600)
    auxiliary = (
        Decimal(AUXILIARY_SECONDS + 6) * (values["cpu_hour_cost"] / 4 + values["mem_gib_hour_cost"]) / Decimal(3600)
    )
    egress = MAX_EGRESS_GIB * values["egress_gib_cost"]
    reservation = (compute + auxiliary + egress + BUILD_AND_RETAINED_STORAGE_ALLOWANCE_USD).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )
    if reservation > MAX_JOB_USD:
        raise RuntimeError(
            f"Live worst-case reservation ${reservation} exceeds per-job cap ${MAX_JOB_USD}; no stage started"
        )
    return {
        "reserved_usd": str(reservation),
        "bounded_compute_usd": str(compute),
        "auxiliary_compute_usd": str(auxiliary),
        "bounded_egress_gib": str(MAX_EGRESS_GIB),
        "egress_allowance_usd": str(egress),
        "build_and_retained_storage_allowance_usd": str(BUILD_AND_RETAINED_STORAGE_ALLOWANCE_USD),
        "resource_spec": spec,
        "input_rates": {key: str(value) for key, value in values.items()},
        "unit_contract": "USD per physical CPU core/L4/GiB-memory hour; egress USD per GiB; 3600 seconds per hour",
        "limitations": "Build/storage allowance is not an absolute lifetime storage bound; failures retain the entire reservation",
    }


def reservation(ledger, run_id, batch_id, stage, revision, manifest_sha, live):
    entries = ledger.get("reservations")
    if not isinstance(entries, list):
        raise RuntimeError("Existing durable course ledger is missing; do not recreate or reset it")
    if any(item.get("run_id") == run_id for item in entries):
        raise ValueError("Every attempt needs a fresh run_id; reservations cannot be reused")
    previous = max(
        Decimal(ledger.get("reserved_total_usd", "0")),
        sum((Decimal(item["reserved_usd"]) for item in entries), Decimal("0")),
    )
    if previous < PRIOR_RESERVED_FLOOR_USD:
        raise RuntimeError("Durable ledger precedes the confirmed $9.84 history; reconcile it before running")
    guard = reservation_guard(stage, live["rates"])
    amount = Decimal(guard["reserved_usd"])
    if previous + amount > TOTAL_CAP_USD:
        raise RuntimeError(
            f"Cumulative reservation {previous} + {amount} exceeds authorized ${TOTAL_CAP_USD}; no stage started"
        )
    item = {
        "run_id": run_id,
        "batch_id": batch_id,
        "experiment_id": f"natural_{stage}",
        "stage": stage,
        "mode": "run" if SPEC[stage]["gpu"] else stage,
        "revision": revision,
        "manifest_sha256": manifest_sha,
        "reserved_usd": str(amount),
        "status": "reserved",
        "billing_before": live,
        "compute_guard": guard,
        "scope": "Natural-assistant extension; prior course reservations preserved; $30 added to original $10 cap",
    }
    entries.append(item)
    ledger.update(budget_usd=str(TOTAL_CAP_USD), reserved_total_usd=str(previous + amount))
    return item


def validate_selection(selection, manifest_sha, adapter_run_id, adapter_checkpoint=""):
    if not isinstance(selection, dict) or selection.get("dataset_manifest_sha256") != manifest_sha:
        raise ValueError("Selected final model must bind the exact frozen dataset manifest SHA-256")
    safe_name(selection.get("validation_run_id", ""))
    if not re.fullmatch(r"[a-f0-9]{64}", selection.get("validation_result_sha256", "")):
        raise ValueError("Selection needs the raw completed validation/result.json SHA-256")
    for field in ("criterion", "decision"):
        if not isinstance(selection.get(field), str) or not selection[field].strip():
            raise ValueError("Final selection must explain its pre-test criterion and decision")
    variant = selection.get("selected_variant")
    if variant == "adapter" or re.fullmatch(r"adapter-step-[0-9]{6}", str(variant)):
        selected_run = safe_name(selection.get("adapter_run_id", ""))
        if adapter_run_id != selected_run or not re.fullmatch(r"[a-f0-9]{64}", selection.get("adapter_sha256", "")):
            raise ValueError("Test adapter argument and immutable weight SHA must match the committed selection")
        checkpoint_names(adapter_checkpoint)
        if selection.get("adapter_checkpoint", "") != adapter_checkpoint or variant != (
            f"adapter-{adapter_checkpoint}" if adapter_checkpoint else "adapter"
        ):
            raise ValueError("Selection must bind the exact archived checkpoint and variant")
    elif variant == "base":
        if (
            adapter_run_id
            or adapter_checkpoint
            or selection.get("adapter_checkpoint")
            or selection.get("adapter_run_id")
            or selection.get("adapter_sha256")
        ):
            raise ValueError("A selected base model must not load an adapter")
    else:
        raise ValueError("Selected variant must be base or adapter")
    return selection


def selection_gate(selection, manifest_sha, adapter_run_id, batch_id, adapter_checkpoint=""):
    validate_selection(selection, manifest_sha, adapter_run_id, adapter_checkpoint)
    report_path = NATURAL_ROOT / batch_id / "validation" / selection["validation_run_id"] / "result.json"
    if not report_path.is_file() or sha256(report_path) != selection["validation_result_sha256"]:
        raise ValueError("Committed selection does not match actual Volume validation/result.json bytes")
    report = json.loads(report_path.read_text())
    execution = report.get("execution", {})
    if (
        report.get("status") != "completed"
        or report.get("split") != "validation"
        or execution.get("manifest_sha256") != manifest_sha
        or execution.get("stage") != "validation"
        or execution.get("run_id") != selection["validation_run_id"]
        or execution.get("batch_id") != batch_id
    ):
        raise ValueError("Selection source must be the completed matching pre-test validation stage")
    variant = selection["selected_variant"]
    if report.get("variants", {}).get(variant, {}).get("completed") is not True:
        raise ValueError("Selected model did not finish the referenced validation evaluation")
    if variant != "base":
        actual = adapter_descriptor(batch_id, adapter_run_id, adapter_checkpoint, manifest_sha)
        evaluated = next((item for item in execution.get("adapters", []) if item.get("variant") == variant), None)
        if evaluated is None and variant == "adapter":
            evaluated = {
                "adapter_run_id": execution.get("adapter_run_id"),
                "adapter_sha256": execution.get("adapter_sha256"),
            }
        if not evaluated or any(evaluated.get(key) != actual[key] for key in ("adapter_run_id", "adapter_sha256")):
            raise ValueError("Selected adapter differs from the weights actually used for validation")
        if actual["adapter_sha256"] != selection["adapter_sha256"]:
            raise ValueError("Selected checkpoint differs from committed inference weights")
    return {
        "validation_run_id": selection["validation_run_id"],
        "validation_result_sha256": selection["validation_result_sha256"],
        "selected_variant": variant,
        "adapter_run_id": selection.get("adapter_run_id"),
        "adapter_sha256": selection.get("adapter_sha256"),
        "adapter_checkpoint": adapter_checkpoint,
        "criterion": selection["criterion"],
        "decision": selection["decision"],
    }


def validate_external_metadata(metadata):
    if metadata.get("schema_version") != 1 or metadata.get("training_allowed") is not False:
        raise ValueError("External OCR metadata must explicitly forbid training")
    rows, files, sources = metadata.get("rows", []), metadata.get("files", []), metadata.get("sources", [])
    if len(rows) != 10 or len(files) != 10 or not sources:
        raise ValueError("External OCR requires the ten preselected frozen image/GT cases")
    for source in sources:
        if (
            not str(source.get("url", "")).startswith("https://")
            or not re.fullmatch(r"[a-f0-9]{40}", source.get("revision", ""))
            or source["revision"] not in source["url"]
            or not re.fullmatch(r"[a-f0-9]{64}", source.get("sha256", ""))
            or type(source.get("bytes")) is not int
            or not 0 < source["bytes"] <= 256 * 1024 * 1024
        ):
            raise ValueError("External TSV source requires exact immutable revision, size and SHA-256")
    for row in rows:
        if row.get("split") != "test" or row.get("task") != "ocr":
            raise ValueError("External cases must stay test-only OCR and cannot select or train the model")
        original_gt = row.get("source", {}).get("original_gt")
        if (
            not isinstance(original_gt, str)
            or row.get("answer") != original_gt
            or hashlib.sha256(original_gt.encode()).hexdigest() != row.get("source", {}).get("original_gt_sha256")
        ):
            raise ValueError("External OCR must retain the unmodified original author GT and its SHA-256")
        safe_relative(row.get("image", ""))
    return metadata


def external_cache_guard(metadata_sha):
    directory = NATURAL_ROOT / "external-ocr" / metadata_sha
    manifest_path = directory / "manifest.json"
    receipt_path = directory / "prepare-receipt.json"
    if not manifest_path.is_file() or sha256(manifest_path) != metadata_sha or not receipt_path.is_file():
        raise ValueError("Run external_prepare on the fixed external metadata before GPU inference")
    metadata = validate_external_metadata(json.loads(manifest_path.read_text()))
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "passed"
        or receipt.get("source_metadata_sha256") != metadata_sha
        or receipt.get("manifest_sha256") != metadata_sha
        or receipt.get("full_tsv_hashes_verified") is not True
        or receipt.get("all_original_gt_and_image_hashes_verified") is not True
        or receipt.get("original_image_count") != 10
    ):
        raise ValueError("External cache lacks the completed original TSV/image/GT verification receipt")
    for item in metadata["files"]:
        path = directory / safe_relative(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Cached external image differs from its pinned original bytes")
    return directory


@app.function(
    image=control_image,
    cpu=(0.25, 0.25),
    memory=(512, 512),
    volumes={"/course": volume},
    timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def reserve_remote(
    run_id,
    batch_id,
    stage,
    revision,
    manifest_sha,
    live,
    selection_text="",
    adapter_run_id="",
    external_metadata_sha="",
    runtime_options_text="",
):
    volume.reload()
    if not LEDGER_PATH.is_file():
        raise RuntimeError("Existing shared course budget.json is absent; refusing a fresh budget")
    ledger = json.loads(LEDGER_PATH.read_text())
    selection_proof = None
    if stage in SELECTION_STAGES:
        settings = json.loads(runtime_options_text) if runtime_options_text else {}
        selection_proof = selection_gate(
            json.loads(selection_text), manifest_sha, adapter_run_id, batch_id, settings.get("adapter_checkpoint", "")
        )
    if stage == "external_ocr":
        external_cache_guard(external_metadata_sha)
    item = reservation(ledger, run_id, batch_id, stage, revision, manifest_sha, live)
    if runtime_options_text:
        item.update(
            runtime_options=json.loads(runtime_options_text),
            runtime_options_sha256=hashlib.sha256(runtime_options_text.encode()).hexdigest(),
        )
    if selection_proof:
        item.update(
            selection_sha256=hashlib.sha256(selection_text.encode()).hexdigest(),
            selection=selection_proof,
        )
    if stage in EXTERNAL_STAGES:
        item["external_metadata_sha256"] = external_metadata_sha
    write_json(LEDGER_PATH, ledger)
    volume.commit()
    return {"budget_usd": ledger["budget_usd"], "reserved_total_usd": ledger["reserved_total_usd"], "entry": item}


def require_reservation(
    run_id, batch_id, stage, revision, manifest_sha, selection_sha="", external_metadata_sha="", runtime_sha=""
):
    ledger = json.loads(LEDGER_PATH.read_text())
    item = next((item for item in ledger["reservations"] if item.get("run_id") == run_id), None)
    if item is None or any(
        item.get(key) != expected
        for key, expected in {
            "batch_id": batch_id,
            "stage": stage,
            "revision": revision,
            "manifest_sha256": manifest_sha,
            "status": "reserved",
        }.items()
    ):
        raise RuntimeError("Stage is missing its exact prior reservation; refusing work")
    if stage in SELECTION_STAGES and item.get("selection_sha256") != selection_sha:
        raise ValueError("Test selection contract differs from the one committed before reserving this attempt")
    if stage in EXTERNAL_STAGES and item.get("external_metadata_sha256") != external_metadata_sha:
        raise ValueError("External OCR metadata differs from the one committed before reserving this attempt")
    if runtime_sha and item.get("runtime_options_sha256") != runtime_sha:
        raise ValueError("Runtime settings differ from the exact configuration reserved before this attempt")
    item["status"] = "running"
    write_json(LEDGER_PATH, ledger)
    volume.commit()
    return item


@app.function(
    image=control_image,
    cpu=(0.25, 0.25),
    memory=(512, 512),
    volumes={"/course": volume},
    timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def finish_remote(run_id, status, live):
    volume.reload()
    ledger = json.loads(LEDGER_PATH.read_text())
    entry = next((item for item in ledger["reservations"] if item.get("run_id") == run_id), None)
    if entry is None:
        raise RuntimeError("Cannot finish an unreserved attempt")
    entry.update(status=status, billing_after=live)
    write_json(LEDGER_PATH, ledger)
    volume.commit()
    return {"budget_usd": ledger["budget_usd"], "reserved_total_usd": ledger["reserved_total_usd"], "entry": entry}


def download_archive(manifest, destination):
    source = manifest.get("source_archive")
    if not isinstance(source, dict):
        raise ValueError("Fixed data manifest needs source_archive with HTTPS URL, SHA-256 and byte length")
    if not str(source.get("url", "")).startswith("https://") or not re.fullmatch(
        r"[a-f0-9]{64}", source.get("sha256", "")
    ):
        raise ValueError("Data archive needs HTTPS and a SHA-256 pin")
    expected_bytes = source.get("bytes")
    if type(expected_bytes) is not int or not 0 < expected_bytes <= 512 * 1024 * 1024:
        raise ValueError("Data archive must declare at most 512 MiB")
    declared_files = source.get("files", [])
    declared = {}
    for item in declared_files:
        relative = safe_relative(item.get("path", "")).as_posix()
        if relative in declared or not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
            raise ValueError("Archive member declarations require unique paths and SHA-256 pins")
        if type(item.get("bytes")) is not int or not 0 <= item["bytes"] <= 2 * 1024 * 1024 * 1024:
            raise ValueError("Archive member byte lengths must be bounded nonnegative integers")
        declared[relative] = item
    receipt = destination / f"archive-{source['sha256']}-receipt.json"
    if receipt.is_file() and json.loads(receipt.read_text()).get("sha256") == source["sha256"]:
        if all(
            (destination / relative).is_file()
            and (destination / relative).stat().st_size == item["bytes"]
            and sha256(destination / relative) == item["sha256"]
            for relative, item in declared.items()
        ):
            return
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix="natural-data-") as temporary:
        archive = Path(temporary) / "data.tar.gz"
        total = 0
        with urllib.request.urlopen(source["url"], timeout=60) as response, archive.open("wb") as stream:
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > expected_bytes:
                    raise ValueError("Archive exceeds declared byte length")
                stream.write(block)
        if total != expected_bytes or sha256(archive) != source["sha256"]:
            raise ValueError("Data archive size or SHA-256 does not match frozen source")
        extracted = Path(temporary) / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if sum(member.size for member in members) > 2 * 1024 * 1024 * 1024:
                raise ValueError("Archive expands beyond the allowed 2 GiB")
            for member in members:
                safe_relative(member.name)
                if not member.isfile() and not member.isdir():
                    raise ValueError("Data archives cannot contain links or special files")
            actual = [safe_relative(member.name).as_posix() for member in members if member.isfile()]
            if len(set(actual)) != len(actual):
                raise ValueError("Archive cannot contain duplicate file entries")
            if declared and set(actual) != set(declared):
                raise ValueError("Archive file allowlist differs from committed per-member declarations")
            tar.extractall(extracted, members=members, filter="data")
        for relative, item in declared.items():
            path = extracted / relative
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError("Extracted archive member bytes differ from the manifest SHA-256 pin")
        for item in extracted.rglob("*"):
            if item.is_file():
                target = destination / item.relative_to(extracted)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, target)
    write_json(receipt, source)


def download_archives(manifest, destination, revision):
    archives = manifest.get("archives", [])
    if not isinstance(archives, list) or not 1 <= len(archives) <= 10:
        raise ValueError("Committed manifest needs a bounded archive list")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Data media URLs require the actual immutable Git checkout SHA")
    if sum(item.get("bytes", 512 * 1024 * 1024 + 1) for item in archives) > 512 * 1024 * 1024:
        raise ValueError("Combined source archives exceed the bounded 512 MiB preparation stage")
    declared_paths = set()
    sources = []
    for archive in archives:
        path = safe_relative(archive.get("path", ""))
        if path.parts[:2] != ("assets", "training") or path.suffixes[-2:] != [".tar", ".gz"]:
            raise ValueError("Data archive must be an assets/training tar.gz LFS asset")
        files = archive.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Every LFS archive requires explicit per-member SHA-256 declarations")
        for item in files:
            name = safe_relative(item.get("path", "")).as_posix()
            if name in declared_paths:
                raise ValueError("Separate archives cannot override each other's declared files")
            declared_paths.add(name)
        source = dict(archive)
        source["url"] = (
            f"https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/{revision}/{path.as_posix()}"
        )
        download_archive({"source_archive": source}, destination)
        sources.append(
            {"path": path.as_posix(), "sha256": source["sha256"], "bytes": source["bytes"], "files": len(files)}
        )
    write_json(destination / "archive-receipt.json", {"git_revision": revision, "archives": sources})


manifest_path = safe_relative(MANIFEST_RELATIVE)
if manifest_path.parts[:2] != ("docs", "natural-assistant"):
    raise ValueError("Manifest must be committed under docs/natural-assistant")
# Remote startup reimports this module without the Actions process environment.
# It must always find prepare_remote/gpu_remote by their top-level names. A
# control-image fallback keeps reserve deployments free of the CUDA build while
# still declaring every entrypoint; unused functions are never invoked there.
natural_image = control_image
if modal.is_local() and PHASE == "execute":
    natural_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
        .pip_install(
            "transformers==4.57.6",
            "peft==0.18.1",
            "accelerate==1.12.0",
            "huggingface-hub==0.36.2",
            "safetensors==0.7.0",
            "numpy==2.2.6",
            "pillow==12.0.0",
            "soundfile==0.13.1",
            "scipy==1.16.3",
        )
        .env(
            {
                "PYTHONPATH": "/app",
                "HF_HOME": "/course/natural-v3/cache/hf",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "4",
                "NATURAL_MANIFEST": MANIFEST_RELATIVE,
            }
        )
        .workdir("/app")
        .add_local_dir(ROOT / "tiny_perceptron", "/app/tiny_perceptron", ignore=["**/__pycache__/**"])
        .add_local_dir(ROOT / "scripts", "/app/scripts", ignore=["**/__pycache__/**"])
        .add_local_file(ROOT / manifest_path, f"/app/{manifest_path.as_posix()}")
        .add_local_file(ROOT / EXTERNAL_METADATA_RELATIVE, f"/app/{EXTERNAL_METADATA_RELATIVE}")
    )


def execute_stage(stage, batch_id, run_id, revision, manifest_sha, options):
    volume.reload()
    runtime_sha = verify_runtime_contract(stage, options)
    require_reservation(
        run_id,
        batch_id,
        stage,
        revision,
        manifest_sha,
        options.get("selection_sha256", ""),
        options.get("external_metadata_sha256", ""),
        runtime_sha,
    )
    selection_proof = None
    if stage in SELECTION_STAGES:
        selection_proof = selection_gate(
            options["selection"], manifest_sha, options["adapter_run_id"], batch_id, options["adapter_checkpoint"]
        )
    manifest_file = Path("/app") / manifest_path
    if sha256(manifest_file) != manifest_sha:
        raise ValueError("Container dataset manifest differs from reserved Git version")
    directory = NATURAL_ROOT / batch_id / stage / run_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_file.read_text())
    data_root = NATURAL_ROOT / "data" / manifest_sha
    external_metadata_sha = options.get("external_metadata_sha256")
    external_root = NATURAL_ROOT / "external-ocr" / external_metadata_sha if external_metadata_sha else None
    if stage in EXTERNAL_STAGES:
        external_file = Path("/app") / EXTERNAL_METADATA_RELATIVE
        if sha256(external_file) != external_metadata_sha:
            raise ValueError("Container external metadata differs from the committed reservation pin")
        validate_external_metadata(json.loads(external_file.read_text()))
    if stage == "external_ocr":
        data_root = external_cache_guard(external_metadata_sha)
        manifest_file = data_root / "manifest.json"
    elif stage == "prepare":
        download_archives(manifest, data_root, revision)
    elif stage != "external_prepare" and not (data_root / "archive-receipt.json").is_file():
        raise RuntimeError("Run the matching CPU prepare stage before any GPU stage")
    args = [
        "python",
        "scripts/natural_assistant.py",
        "evaluate" if stage == "external_ocr" else stage,
        "--manifest",
        str(manifest_file),
        "--data-root",
        str(data_root),
        "--output",
        str(directory),
        "--cache-dir",
        str(NATURAL_ROOT / "cache" / "models"),
        "--device",
        "cpu" if stage == "prepare" else "cuda",
        "--dtype",
        "bfloat16",
        "--max-seconds",
        str(options["max_seconds"]),
        "--max-pixels",
        str(options["max_pixels"]),
        "--seed",
        str(options["seed"]),
        "--asr-model",
        options["asr_model"],
        "--asr-revision",
        options["asr_revision"],
    ]
    if stage != "prepare":
        args.append("--local-files-only")
    if stage in ("baseline", "validation", "evaluate"):
        args.extend(["--split", "test" if stage == "evaluate" else "validation", "--max-new-tokens", "384"])
    if stage == "evaluate":
        args.append("--selected-only")
    if stage == "train":
        args.extend(
            [
                "--steps",
                str(options["steps"]),
                "--checkpoint-every",
                "25",
                "--learning-rate",
                str(options["learning_rate"]),
            ]
        )
        if options["checkpoint_steps"]:
            args.extend(["--checkpoint-steps", ",".join(map(str, options["checkpoint_steps"]))])
    if stage == "external_ocr":
        args.extend(["--split", "test", "--max-new-tokens", "384"])
    if stage == "external_prepare":
        args = [
            "python",
            "scripts/prepare_external_ocr.py",
            "--metadata",
            str(external_file),
            "--output",
            str(external_root),
            "--cache-dir",
            str(NATURAL_ROOT / "cache" / "external-ocr"),
        ]
    adapter_sha = None
    adapters = []
    if options.get("adapter_run_id") and stage != "external_prepare":
        adapter_run_id = safe_name(options["adapter_run_id"])
        checkpoints = options["adapter_checkpoints"] or [options["adapter_checkpoint"]]
        adapters = [
            adapter_descriptor(batch_id, adapter_run_id, checkpoint, manifest_sha) for checkpoint in checkpoints
        ]
        adapter_sha = adapters[0]["adapter_sha256"]
        args.extend(["--adapter", adapters[0]["path"], "--adapter-label", adapters[0]["variant"]])
        if options["adapter_checkpoints"]:
            for item in adapters:
                args.extend(["--comparison-adapter", f"{item['variant']}={item['path']}"])
    write_json(
        directory / "execution.json",
        {
            "stage": stage,
            "batch_id": batch_id,
            "run_id": run_id,
            "revision": revision,
            "manifest_sha256": manifest_sha,
            "runner_arguments": args,
            "resource_spec": SPEC[stage],
            "secret_injected_into_gpu": False,
            "adapter_run_id": options.get("adapter_run_id") or None,
            "adapter_sha256": adapter_sha,
            "adapter_checkpoint": options["adapter_checkpoint"],
            "adapters": adapters,
            "runtime_options_sha256": options["runtime_options_sha256"],
            "runtime_options": options["runtime_options"],
            "selection_sha256": options.get("selection_sha256"),
            "pretest_selection": selection_proof,
            "external_metadata_sha256": external_metadata_sha,
            "external_evaluation_only": stage in EXTERNAL_STAGES,
        },
    )
    volume.commit()
    stop = threading.Event()
    commits = []

    def persist():
        while not stop.wait(60):
            try:
                volume.commit()
            except Exception as error:
                commits.append(type(error).__name__)

    persistence = threading.Thread(target=persist, daemon=True)
    persistence.start()
    started = time.monotonic()
    try:
        with (directory / "runner.log").open("w", encoding="utf-8") as log:
            subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=options["max_seconds"] + 120)
        result_path = directory / "result.json"
        if stage == "external_prepare":
            external_cache_guard(external_metadata_sha)
            receipt = json.loads((external_root / "prepare-receipt.json").read_text())
            write_json(result_path, {"status": "completed", "external_preparation": receipt})
        if not result_path.is_file():
            raise RuntimeError("Runner finished without result.json; do not mark the experiment complete")
        result = json.loads(result_path.read_text())
        result["execution"] = {
            "run_id": run_id,
            "batch_id": batch_id,
            "revision": revision,
            "manifest_sha256": manifest_sha,
            "stage": stage,
            "seconds": time.monotonic() - started,
            "volume_commit_errors": commits,
            "adapter_run_id": options.get("adapter_run_id") or None,
            "adapter_sha256": adapter_sha,
            "adapter_checkpoint": options["adapter_checkpoint"],
            "adapters": adapters,
            "runtime_options_sha256": options["runtime_options_sha256"],
            "runtime_options": options["runtime_options"],
            "selection_sha256": options.get("selection_sha256"),
            "pretest_selection": selection_proof,
            "external_metadata_sha256": external_metadata_sha,
            "external_evaluation_only": stage in EXTERNAL_STAGES,
        }
        if stage == "external_ocr":
            from PIL import Image

            external_metadata = json.loads(manifest_file.read_text())
            evidence = []
            for row in external_metadata["rows"]:
                image_path = data_root / safe_relative(row["image"])
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
                    pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
                    size = list(image.size)
                evidence.append(
                    {
                        "id": row["id"],
                        "original_gt": row["source"]["original_gt"],
                        "original_gt_sha256": row["source"]["original_gt_sha256"],
                        "original_image_sha256": sha256(image_path),
                        "original_image_size": size,
                        "decoded_rgb_pixel_sha256": pixel_sha,
                        "pixel_hash_scope": "Decoded original RGB pixels before AutoProcessor resize/normalization; not model tensor values",
                        "annotation_review": row["source"].get("annotation_review"),
                    }
                )
            result["external_evaluation"] = {
                "metadata_sha256": external_metadata_sha,
                "selected_before_test_sha256": options["selection_sha256"],
                "max_new_tokens": 384,
                "max_pixels": options["max_pixels"],
                "training_performed": False,
                "raw_images_redistributed": False,
                "rows": evidence,
                "scoring_policy": external_metadata["scoring_policy"],
                "split_policy": external_metadata["split_policy"],
            }
            write_json(directory / "external-evidence.json", result["external_evaluation"])
        write_json(result_path, result)
        return result
    except Exception as error:
        write_json(
            directory / "failure.json",
            {"run_id": run_id, "revision": revision, "exception_type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        stop.set()
        persistence.join(timeout=5)
        volume.commit()


@app.function(
    image=natural_image,
    cpu=(2, 2),
    memory=(8192, 8192),
    volumes={"/course": volume},
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def prepare_remote(batch_id, run_id, revision, manifest_sha, options):
    return execute_stage("prepare", batch_id, run_id, revision, manifest_sha, options)


@app.function(
    image=natural_image,
    cpu=(2, 2),
    memory=(8192, 8192),
    volumes={"/course": volume},
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def external_prepare_remote(batch_id, run_id, revision, manifest_sha, options):
    return execute_stage("external_prepare", batch_id, run_id, revision, manifest_sha, options)


@app.function(
    image=natural_image,
    gpu="L4",
    cpu=(4, 4),
    memory=(32768, 32768),
    volumes={"/course": volume},
    timeout=3600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def gpu_remote(stage, batch_id, run_id, revision, manifest_sha, options):
    if stage not in ("baseline", "train", "validation", "evaluate", "external_ocr"):
        raise ValueError("Only fixed GPU stages are allowed")
    return execute_stage(stage, batch_id, run_id, revision, manifest_sha, options)


@app.function(
    image=control_image,
    cpu=(0.25, 0.25),
    memory=(1024, 1024),
    secrets=[hf_secret],
    volumes={"/course": volume},
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def backup_remote(checkpoint_repo, batch_id, stage, run_id, revision):
    from huggingface_hub import HfApi, hf_hub_download

    volume.reload()
    directory = NATURAL_ROOT / batch_id / stage / run_id
    directory.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=os.environ["HF_TOKEN"])
    if not api.repo_info(checkpoint_repo, repo_type="model").private:
        raise ValueError("Full training backup repository must be private")
    files = [
        {"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink() and not path.name.endswith(".tmp")
    ]
    if sum(item["bytes"] for item in files) > MAX_PRIVATE_BACKUP_BYTES:
        raise ValueError("Private stage backup exceeds the reserved 256 MiB; narrow checkpoint retention first")
    write_json(directory / "upload-manifest.json", {"files": files, "revision": revision, "run_id": run_id})
    prefix = f"natural-v3/{batch_id}/{stage}/{run_id}"
    commit = api.upload_folder(
        repo_id=checkpoint_repo,
        folder_path=str(directory),
        path_in_repo=prefix,
        ignore_patterns=["*.tmp", "**/*.tmp", "**/.cache/**"],
        commit_message=f"Save complete natural-assistant {stage}: {run_id}",
    )
    verified = []
    with tempfile.TemporaryDirectory(prefix="natural-private-check-") as temporary:
        for item in files:
            if Path(item["path"]).suffix not in (".safetensors", ".pt", ".pth"):
                continue
            downloaded = hf_hub_download(
                repo_id=checkpoint_repo,
                filename=f"{prefix}/{item['path']}",
                revision=commit.oid,
                token=os.environ["HF_TOKEN"],
                local_dir=temporary,
            )
            if sha256(downloaded) != item["sha256"]:
                raise RuntimeError("Private HF weight bytes differ from the trained checkpoint")
            verified.append(item["path"])
    result = {
        "repo": checkpoint_repo,
        "revision": commit.oid,
        "prefix": prefix,
        "files": files,
        "verified_weight_files": verified,
        "private": True,
        "resume_scope": "Exact optimizer/RNG continuation requires runner checkpoint support; adapter files alone are inference weights",
    }
    write_json(directory / "hf-receipt.json", result)
    volume.commit()
    return result


def validate_release(approval):
    if not isinstance(approval, dict) or approval.get("approved") is not True or approval.get("reviewed") is not True:
        raise ValueError("Release needs a concrete reviewed, committed model/file approval")
    variant = approval.get("selected_variant", "adapter")
    if not isinstance(variant, str) or (
        variant not in {"base", "adapter"} and not re.fullmatch(r"adapter-step-[0-9]{6}", variant)
    ):
        raise ValueError("Release must identify base or one reviewed adapter variant")
    base_only = variant == "base"
    source = approval.get("source")
    if base_only:
        if source is not None or (
            "adapter_parameters" in approval
            and (type(approval["adapter_parameters"]) is not int or approval["adapter_parameters"] != 0)
        ):
            raise ValueError("Base-only release cannot claim private adapter weights or parameters")
    else:
        if not isinstance(source, dict) or not re.fullmatch(r"[a-f0-9]{40}", source.get("revision", "")):
            raise ValueError("Private source must use an immutable HF commit")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source.get("repo", "")):
            raise ValueError("Expected private source owner/repository")
        prefix = safe_relative(source.get("prefix", ""))
        if prefix.parts[0] != "natural-v3":
            raise ValueError("Release source must belong to this versioned extension")
    files = approval.get("files")
    if (
        not isinstance(files, list)
        or any(not isinstance(item, dict) for item in files)
        or (bool(files) if base_only else not files)
        or len(files) > 50
    ):
        raise ValueError("Release needs a bounded explicit file allowlist")
    if any(type(item.get("bytes")) is not int or not 0 < item["bytes"] <= MAX_PRIVATE_BACKUP_BYTES for item in files):
        raise ValueError("Every released file needs a positive bounded integer byte length")
    if sum(item.get("bytes", MAX_PRIVATE_BACKUP_BYTES + 1) for item in files) > MAX_PRIVATE_BACKUP_BYTES:
        raise ValueError("Public adapter export exceeds its reserved size")
    if len({item.get("path") for item in files}) != len(files):
        raise ValueError("Public release file paths must be unique")
    for item in files:
        path = safe_relative(item.get("path", ""))
        if path.suffix not in (".safetensors", ".json", ".md") or any(
            word in part.lower()
            for part in path.parts
            for word in ("optimizer", "scheduler", "rng", "checkpoint", "training")
        ):
            raise ValueError("Only reviewed inference adapter/config/metadata files may be public")
        if "source_path" in item:
            private_path = safe_relative(item["source_path"])
            if private_path.name != path.name or private_path.name not in (
                "adapter_model.safetensors",
                "adapter_config.json",
            ):
                raise ValueError("Checkpoint mapping may export only the same inference adapter/config basename")
            if not (
                private_path.parts == ("adapter", private_path.name)
                or len(private_path.parts) == 3
                and private_path.parts[0] == "checkpoints"
                and len(checkpoint_names(private_path.parts[1])) == 1
            ):
                raise ValueError("Checkpoint mapping must identify latest or one explicit archived adapter")
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")):
            raise ValueError("Every released file needs an exact SHA-256")
        if item.get("redistribution_approved") is not True or not item.get("license"):
            raise ValueError("Every released file needs an explicit redistribution license")
    if not isinstance(approval.get("model_card"), str) or not approval["model_card"].strip():
        raise ValueError("Release needs the actual reviewed model card")
    for key in ("base_model", "asr_model"):
        model = approval.get(key, {})
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model.get("repo", "")) or not re.fullmatch(
            r"[a-f0-9]{40}", model.get("revision", "")
        ):
            raise ValueError("Public model and speech route require explicit immutable pretrained model pins")
    safe_name(approval.get("release_id", ""))
    return approval


@app.function(
    image=control_image,
    cpu=(1, 1),
    memory=(4096, 4096),
    secrets=[hf_secret],
    volumes={"/course": volume},
    timeout=1800,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def release_remote(release_repo, approval_text, approval_sha, run_id, batch_id, revision, manifest_sha):
    from huggingface_hub import HfApi, hf_hub_download

    volume.reload()
    require_reservation(run_id, batch_id, "release", revision, manifest_sha)
    if hashlib.sha256(approval_text.encode()).hexdigest() != approval_sha:
        raise ValueError("Committed review bytes differ from the Actions approval hash")
    approval = validate_release(json.loads(approval_text))
    api = HfApi(token=os.environ["HF_TOKEN"])
    source = approval.get("source")
    if source is not None and not api.repo_info(source["repo"], repo_type="model").private:
        raise ValueError("Reviewed training source must be private")
    if api.repo_info(release_repo, repo_type="model").private:
        raise ValueError("Student release repository must be public")
    for key in ("base_model", "asr_model"):
        pinned = approval[key]
        info = HfApi(token=False).model_info(pinned["repo"], revision=pinned["revision"])
        if info.private or info.sha != pinned["revision"]:
            raise ValueError("A pinned pretrained dependency is not publicly accessible at the stated commit")
    prefix = f"natural-v3/{approval['release_id']}"
    with tempfile.TemporaryDirectory(prefix="natural-release-") as temporary:
        export = Path(temporary) / "export"
        export.mkdir()
        for item in approval["files"]:
            file = hf_hub_download(
                repo_id=source["repo"],
                filename=f"{source['prefix']}/{item.get('source_path', item['path'])}",
                revision=source["revision"],
                token=os.environ["HF_TOKEN"],
                local_dir=Path(temporary) / "private",
            )
            if Path(file).stat().st_size != item["bytes"] or sha256(file) != item["sha256"]:
                raise ValueError("Reviewed private inference bytes changed")
            target = export / safe_relative(item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
        (export / "README.md").write_text(approval["model_card"], encoding="utf-8")
        provenance = {
            "approval_sha256": approval_sha,
            "git_revision": revision,
            "manifest_sha256": manifest_sha,
            "source": source,
            "base_model": approval["base_model"],
            "asr_model": approval["asr_model"],
        }
        if "selected_variant" in approval:
            provenance["selected_variant"] = approval["selected_variant"]
        write_json(export / "release-provenance.json", provenance)
        commit = api.upload_folder(
            repo_id=release_repo,
            folder_path=str(export),
            path_in_repo=prefix,
            commit_message=f"Publish reviewed natural-assistant {approval.get('selected_variant', 'adapter')}: {run_id}",
        )
        files = [
            {"path": f"{prefix}/{p.relative_to(export).as_posix()}", "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(export.rglob("*"))
            if p.is_file()
        ]
        for item in files:
            file = hf_hub_download(
                repo_id=release_repo,
                filename=item["path"],
                revision=commit.oid,
                token=False,
                local_dir=Path(temporary) / "anonymous",
            )
            if Path(file).stat().st_size != item["bytes"] or sha256(file) != item["sha256"]:
                raise RuntimeError("Anonymous student download differs from approved export")
    result = {"repo": release_repo, "revision": commit.oid, "files": files, "anonymous_download_verified": True}
    if "selected_variant" in approval:
        result["selected_variant"] = approval["selected_variant"]
    directory = NATURAL_ROOT / batch_id / "release" / run_id
    write_json(directory / "result.json", result)
    volume.commit()
    return result


def committed_text(relative, revision, prefix):
    path = safe_relative(relative)
    if path.parts[: len(prefix)] != tuple(prefix):
        raise ValueError("Input must be inside the committed extension manifest/release directory")
    content = subprocess.run(
        ["git", "show", f"{revision}:{path.as_posix()}"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    if (ROOT / path).read_bytes() != content:
        raise ValueError("Input differs from the selected committed Git revision")
    return content.decode("utf-8")


def download_review(batch_id, stage, run_id, output, receipt):
    allowed = {
        "result.json",
        "generations.json",
        "generations-base.json",
        "generations-adapter.json",
        "transcripts.json",
        "external-evidence.json",
        "provenance.json",
        "training.json",
        "execution.json",
        "failure.json",
        "runner.log",
    }
    total = 0
    for item in receipt["files"]:
        relative = safe_relative(item["path"])
        adapter_file = relative.parts[0] == "adapter" and relative.suffix in (".json", ".safetensors")
        archived_file = (
            len(relative.parts) == 3
            and relative.parts[0] == "checkpoints"
            and bool(checkpoint_names(relative.parts[1]))
            and relative.suffix in (".json", ".safetensors")
        )
        candidate_output = len(relative.parts) == 1 and re.fullmatch(
            r"generations-adapter-step-[0-9]{6}\.json", relative.name
        )
        if relative.as_posix() not in allowed and not (adapter_file or archived_file or candidate_output):
            continue
        total += item["bytes"]
        if total > MAX_REVIEW_DOWNLOAD_BYTES:
            raise ValueError("Review artifact exceeds reserved 64 MiB; keep fixed private HF receipt instead")
        target = output / "review" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            for block in volume.read_file(f"natural-v3/{batch_id}/{stage}/{run_id}/{relative.as_posix()}"):
                stream.write(block)
        if sha256(target) != item["sha256"] or target.stat().st_size != item["bytes"]:
            raise RuntimeError("Actions review copy differs from private-backup receipt")


@app.local_entrypoint()
def main(
    stage: str,
    run_id: str,
    revision: str,
    batch_id: str = "natural-v3",
    requested_stage: str = "",
    manifest: str = "docs/natural-assistant/manifest.json",
    checkpoint_repo: str = "",
    release_repo: str = "",
    approval_file: str = "",
    selection_file: str = "docs/natural-assistant/selection.json",
    adapter_run_id: str = "",
    steps: int = 100,
    seed: int = 42,
    max_seconds: int = 3300,
    max_pixels: int = 524288,
    learning_rate: float = 0.00003,
    asr_variant: str = "small",
    checkpoint_steps: str = "",
    adapter_checkpoint: str = "",
    adapter_checkpoints: str = "",
    finish_status: str = "failed-or-cancelled",
):
    for value in (run_id, batch_id):
        safe_name(value)
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Need actual full Git checkout SHA")
    if stage not in (*STAGES, "reserve", "finish"):
        raise ValueError("Unknown stage")
    selected = requested_stage if stage == "reserve" else stage
    if stage == "reserve" and selected not in STAGES:
        raise ValueError("Reserve must specify the exact next stage")
    if selected == "validation":
        if not adapter_run_id:
            raise ValueError("Post-training validation needs the exact train adapter_run_id before model selection")
        safe_name(adapter_run_id)
    output = ROOT / "outputs" / "natural-assistant"
    if stage == "finish":
        result = finish_remote.remote(run_id, finish_status, snapshot())
        write_json(output / "budget-final.json", result)
        return
    settings = runtime_options(
        selected,
        steps,
        seed,
        max_seconds,
        max_pixels,
        learning_rate,
        asr_variant,
        checkpoint_steps,
        adapter_checkpoint,
        adapter_checkpoints,
    )
    if (adapter_checkpoint or adapter_checkpoints) and not adapter_run_id:
        raise ValueError("Archived checkpoint selection needs its exact train adapter_run_id")
    settings_text = runtime_contract(settings)
    manifest_text = committed_text(manifest, revision, ("docs", "natural-assistant"))
    manifest_sha = hashlib.sha256(manifest_text.encode()).hexdigest()
    selection_text = ""
    if selected in SELECTION_STAGES:
        selection_text = committed_text(selection_file, revision, ("docs", "natural-assistant"))
        validate_selection(json.loads(selection_text), manifest_sha, adapter_run_id, adapter_checkpoint)
    external_metadata_sha = ""
    if selected in EXTERNAL_STAGES:
        external_text = committed_text(EXTERNAL_METADATA_RELATIVE, revision, ("docs", "natural-assistant"))
        validate_external_metadata(json.loads(external_text))
        external_metadata_sha = hashlib.sha256(external_text.encode()).hexdigest()
    if stage == "reserve":
        approval_text = ""
        if selected == "release":
            approval_text = committed_text(approval_file, revision, ("docs", "natural-assistant", "releases"))
            validate_release(json.loads(approval_text))
        live = snapshot()
        # Local validation precedes even the inexpensive ledger container call.
        reservation_guard(selected, live["rates"])
        result = reserve_remote.remote(
            run_id,
            batch_id,
            selected,
            revision,
            manifest_sha,
            live,
            selection_text,
            adapter_run_id,
            external_metadata_sha,
            settings_text,
        )
        write_json(output / "reservation.json", result)
        print(
            json.dumps(
                {
                    "stage": selected,
                    "reserved_usd": result["entry"]["reserved_usd"],
                    "cumulative_reserved_usd": result["reserved_total_usd"],
                }
            )
        )
        return
    if stage != "release" and PHASE != "execute":
        raise RuntimeError("Heavy stages require NATURAL_MODAL_PHASE=execute after a separate reservation")
    options = {
        **settings,
        "runtime_options": settings,
        "runtime_options_sha256": hashlib.sha256(settings_text.encode()).hexdigest(),
        "adapter_run_id": adapter_run_id,
        "selection": json.loads(selection_text) if selection_text else None,
        "selection_sha256": hashlib.sha256(selection_text.encode()).hexdigest() if selection_text else None,
        "external_metadata_sha256": external_metadata_sha or None,
    }
    result = {"stage": stage, "run_id": run_id, "revision": revision}
    error = None
    try:
        if stage == "release":
            text = committed_text(approval_file, revision, ("docs", "natural-assistant", "releases"))
            result["release"] = release_remote.remote(
                release_repo, text, hashlib.sha256(text.encode()).hexdigest(), run_id, batch_id, revision, manifest_sha
            )
        else:
            try:
                if stage == "prepare":
                    result["experiment"] = prepare_remote.remote(batch_id, run_id, revision, manifest_sha, options)
                elif stage == "external_prepare":
                    result["experiment"] = external_prepare_remote.remote(
                        batch_id, run_id, revision, manifest_sha, options
                    )
                else:
                    result["experiment"] = gpu_remote.remote(stage, batch_id, run_id, revision, manifest_sha, options)
            finally:
                result["hf"] = backup_remote.remote(checkpoint_repo, batch_id, stage, run_id, revision)
                download_review(batch_id, stage, run_id, output, result["hf"])
        result["status"] = result.get("experiment", {}).get("status", "completed")
    except Exception as failure:
        error = failure
        result.update(status="failed", exception_type=type(failure).__name__, message=str(failure))
    finally:
        try:
            result["budget"] = finish_remote.remote(run_id, "failed" if error else result["status"], snapshot())
        except Exception as failure:
            result["budget_error"] = {"exception_type": type(failure).__name__, "message": str(failure)}
            error = error or failure
        write_json(output / "result.json", result)
        print(json.dumps({"stage": stage, "status": result.get("status"), "run_id": run_id}, ensure_ascii=False))
    if error:
        raise RuntimeError(
            "Bounded experiment failed; reservation retained and partial output backed up where possible"
        ) from error
