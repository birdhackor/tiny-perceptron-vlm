"""Read completed capstone inference artifacts for review, without starting compute.

Uses the public Modal 1.6.0 Volume.from_name/read_file API and the client's normal
authentication. This working review transport does not approve student releases.
It never reads training checkpoints, billing, secrets, or an arbitrary glob.
"""

import argparse
import hashlib
import importlib.metadata
import json
import pickletools
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

VOLUME_NAME = "tiny-perceptron-course"
ALLOWED_BATCHES = frozenset({"course-v1", "course-integration-v2"})
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
STAGE_FILES = frozenset({"model.pt", "data-manifest.json", "train-report.json", "validation.json"})
DEPLOYMENT_FILES = frozenset(
    {
        "model.pt",
        "model-int4.pt",
        "pretrain.pt",
        "sft.pt",
        "joint.pt",
        "dpo.pt",
        "dpo-int4.pt",
        "dpo-int8.pt",
        "joint-int4.pt",
        "joint-int8.pt",
        "data.json",
        "test-untrained.json",
        "test-pretrain.json",
        "test-sft.json",
        "test-joint.json",
        "test-dpo.json",
        "test-ptq4.json",
        "test-ptq8.json",
        "test-audio-swaps.json",
        "test-image-swaps.json",
        "test-image-pairs.json",
        "test-dpo-audio-swaps.json",
        "test-joint-audio-swaps.json",
        "test-joint-image-swaps.json",
        "test-joint-image-pairs.json",
        "test-joint-ptq4.json",
        "test-joint-ptq8.json",
        "cache-consistency.json",
        "mechanism-benchmark.json",
        "generation-benchmark.json",
        "deployment-report.json",
    }
)
STUDENT_FILES = frozenset(
    {
        "ce/model.pt",
        "kd/model.pt",
        "model.pt",
        "model-int4.pt",
        "ce/train-report.json",
        "kd/train-report.json",
        "validation-ce.json",
        "validation-kd.json",
        "test-ce.json",
        "test-kd.json",
        "test-kd-ptq4.json",
        "student-report.json",
    }
)
ALLOWLIST = {
    "capstone_pretrain": STAGE_FILES,
    "capstone_sft": STAGE_FILES,
    "capstone_joint": STAGE_FILES,
    "capstone_preference": STAGE_FILES,
    "capstone_deployment": DEPLOYMENT_FILES,
    "capstone_student": STUDENT_FILES,
}
REQUIRED = {
    **{name: STAGE_FILES for name in ("capstone_pretrain", "capstone_sft", "capstone_joint", "capstone_preference")},
    "capstone_deployment": frozenset(
        {
            "pretrain.pt",
            "sft.pt",
            "joint.pt",
            "dpo.pt",
            "dpo-int4.pt",
            "dpo-int8.pt",
            "joint-int4.pt",
            "joint-int8.pt",
            "data.json",
            "deployment-report.json",
        }
    ),
    "capstone_student": frozenset({"ce/model.pt", "kd/model.pt", "model.pt", "model-int4.pt", "student-report.json"}),
}
FORBIDDEN_FIELDS = frozenset(
    {
        "optimizer",
        "reference",
        "training_state",
        "torch_rng",
        "python_rng",
        "cuda_rng",
        "rng",
        "billing",
        "secret",
        "secrets",
        "password",
        "token",
        "access_token",
        "hf_token",
        "modal_token_id",
        "modal_token_secret",
    }
)


class ReviewDownloadError(ValueError):
    """A safe diagnostic that never contains credentials or signed URLs."""


def _relative(value):
    if not isinstance(value, str):
        raise ReviewDownloadError("Artifact paths must be normalized relative strings")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or path.as_posix() != value:
        raise ReviewDownloadError("Artifact paths must be normalized relative strings")
    return path


def _identity(result, experiment, batch):
    if (
        result.get("schema_version") != 1
        or result.get("experiment_id") != experiment
        or result.get("status") != "completed"
        or result.get("evidence_status") != "complete_run"
        or not re.fullmatch(r"[0-9a-f]{40}", result.get("revision", ""))
        or result.get("modal", {}).get("batch_id") != batch
        or not re.fullmatch(r"gha-[1-9][0-9]*-[1-9][0-9]*", result.get("modal", {}).get("run_id", ""))
    ):
        raise ReviewDownloadError(
            "Review transport requires the matching complete formal Actions experiment/run/revision"
        )
    return result["revision"], result["modal"]["run_id"]


def _artifacts(result):
    rows = result.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ReviewDownloadError("Completed result must contain its actual artifact manifest")
    artifacts = {}
    for row in rows:
        name = _relative(row.get("path")).as_posix()
        if (
            name in artifacts
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
        ):
            raise ReviewDownloadError("Artifact manifest requires unique paths and exact size/SHA-256")
        artifacts[name] = {key: row[key] for key in ("path", "bytes", "sha256")}
    return artifacts


def _chunks(volume, path):
    try:
        yield from volume.read_file(path)
    except Exception:
        # SDK/network exceptions can contain signed URLs; expose only a safe hint.
        raise ReviewDownloadError(
            "Modal Volume read failed; check official client authentication, volume access and completed run"
        ) from None


def _read_result(volume, path):
    data = bytearray()
    for chunk in _chunks(volume, path):
        if not isinstance(chunk, bytes) or len(data) + len(chunk) > MAX_FILE_BYTES:
            raise ReviewDownloadError("Volume result exceeds the per-file streaming limit")
        data.extend(chunk)
    try:
        result = json.loads(data)
    except (UnicodeError, ValueError):
        raise ReviewDownloadError("Volume result is not a valid JSON object") from None
    if not isinstance(result, dict):
        raise ReviewDownloadError("Volume result must be a JSON object")
    return result, len(data)


def _reject_private_fields(value):
    if isinstance(value, dict):
        if value.get("private_only") is True or any(str(key).lower() in FORBIDDEN_FIELDS for key in value):
            raise ReviewDownloadError("Review JSON contains private training, billing or authentication fields")
        for item in value.values():
            _reject_private_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private_fields(item)


def _inference_archive(path):
    """Inspect PyTorch pickle opcodes without executing pickle or importing torch.

    The Actions controller installs only Modal. Exact source hashes plus an
    explicit inference flag and rejected training fields gate this transport;
    strict model reconstruction remains a separate release/CPU review check.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            records = archive.infolist()
            pickles = [record for record in records if record.filename.endswith("/data.pkl")]
            if len(pickles) != 1 or sum(record.file_size for record in records) > MAX_FILE_BYTES:
                raise ReviewDownloadError("Inference checkpoint is not a bounded PyTorch archive")
            operations = list(pickletools.genops(archive.read(pickles[0])))
    except (OSError, zipfile.BadZipFile, ValueError):
        raise ReviewDownloadError("Inference checkpoint archive cannot be verified without executing pickle") from None
    string_ops = {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}
    strings = [argument for operation, argument, _ in operations if operation.name in string_ops]
    if any(value.lower() in FORBIDDEN_FIELDS for value in strings):
        raise ReviewDownloadError("Checkpoint contains forbidden training or authentication fields")
    if strings.count("format_version") != 1 or strings.count("inference_only") != 1:
        raise ReviewDownloadError("Checkpoint requires one explicit format and inference-only declaration")
    declared = {}
    memo_ops = {"PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"}
    for index, (operation, argument, _) in enumerate(operations):
        if operation.name not in string_ops or argument not in ("format_version", "inference_only"):
            continue
        following = index + 1
        while following < len(operations) and operations[following][0].name in memo_ops:
            following += 1
        if following >= len(operations):
            raise ReviewDownloadError("Inference declaration is incomplete")
        next_operation, value, _ = operations[following]
        declared[argument] = (next_operation.name, value)
    if (
        declared.get("format_version", (None, None))[0] not in string_ops
        or declared["format_version"][1] not in ("capstone-v1", "capstone-ptq-v1")
        or declared.get("inference_only") != ("NEWTRUE", None)
    ):
        raise ReviewDownloadError("Only explicitly stripped capstone inference checkpoints can enter review artifacts")


def download_review(result_path, output, *, experiment=None, batch=None, volume=None):
    result_path = Path(result_path)
    if result_path.stat().st_size > MAX_FILE_BYTES:
        raise ReviewDownloadError("Local Actions result exceeds the per-file limit")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    experiment = experiment or result.get("experiment_id")
    batch = batch or result.get("modal", {}).get("batch_id")
    if (
        experiment not in ALLOWLIST
        or batch not in ALLOWED_BATCHES
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", batch)
        or ".." in batch
    ):
        raise ReviewDownloadError("Choose an explicitly supported capstone experiment and allowlisted safe batch")
    revision, run_id = _identity(result, experiment, batch)
    expected = _artifacts(result)
    selected = {name: row for name, row in expected.items() if name in ALLOWLIST[experiment]}
    if not REQUIRED[experiment] <= selected.keys():
        raise ReviewDownloadError("Completed capstone result lacks required inference review files")
    if any(not 0 < row["bytes"] <= MAX_FILE_BYTES for row in selected.values()):
        raise ReviewDownloadError("A selected review file exceeds the 15 MiB per-file limit")
    declared_bytes = sum(row["bytes"] for row in selected.values())
    if declared_bytes > MAX_TOTAL_BYTES:
        raise ReviewDownloadError("Selected review files exceed the 64 MiB total streaming limit")
    if volume is None:
        try:
            import modal
        except ImportError:
            raise ReviewDownloadError(
                "Modal SDK is unavailable; install modal==1.6.0 and use official Modal authentication"
            ) from None
        if importlib.metadata.version("modal") != "1.6.0":
            raise ReviewDownloadError("This review transport is verified against modal==1.6.0")
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    prefix = f"{batch}/{experiment}"
    remote, result_bytes = _read_result(volume, f"{prefix}/result.json")
    if _identity(remote, experiment, batch) != (revision, run_id) or _artifacts(remote) != expected:
        raise ReviewDownloadError(
            "Volume run/revision/artifacts differ from the anchored Actions result; do not mix runs"
        )
    if declared_bytes + result_bytes > MAX_TOTAL_BYTES:
        raise ReviewDownloadError("Result and selected files exceed the 64 MiB total streaming limit")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / experiment / run_id
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise ReviewDownloadError("Review destination already exists; choose a fresh output directory")
    verified = []
    streamed = result_bytes
    with tempfile.TemporaryDirectory(prefix=".capstone-review-", dir=target.parent) as temporary:
        directory = Path(temporary)
        for name, row in sorted(selected.items()):
            destination = directory / _relative(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with destination.open("wb") as stream:
                for chunk in _chunks(volume, f"{prefix}/{name}"):
                    if not isinstance(chunk, bytes):
                        raise ReviewDownloadError("Modal file stream yielded a non-byte chunk")
                    count += len(chunk)
                    streamed += len(chunk)
                    if count > min(row["bytes"], MAX_FILE_BYTES) or streamed > MAX_TOTAL_BYTES:
                        raise ReviewDownloadError("Modal stream exceeds declared size or review download limits")
                    digest.update(chunk)
                    stream.write(chunk)
            if count != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ReviewDownloadError("Review file differs from its anchored result size/SHA-256")
            if destination.suffix == ".pt":
                _inference_archive(destination)
            else:
                try:
                    _reject_private_fields(json.loads(destination.read_text(encoding="utf-8")))
                except (UnicodeError, ValueError) as error:
                    if isinstance(error, ReviewDownloadError):
                        raise
                    raise ReviewDownloadError("Review JSON cannot be verified") from None
            verified.append(row)
        receipt = {
            "schema_version": 1,
            "purpose": "Working review transport; no student publication approval",
            "experiment_id": experiment,
            "batch_id": batch,
            "run_id": run_id,
            "training_revision": revision,
            "source": {"api": "modal.Volume.from_name/read_file", "volume": VOLUME_NAME, "prefix": prefix},
            "sdk": {
                "version": "1.6.0",
                "reference": "https://modal.com/docs/reference/modal.Volume#read_file",
                "verified_distribution_sha256": "2f5981ba656570ad96379d29f0ddff876880708c84966024ce130e9e1794fa68",
            },
            "streamed_bytes_including_result": streamed,
            "files": verified,
            "checkpoint_check_scope": "SHA/size and non-executing pickle inspection only; CPU reconstruction and strict export review still required before student publication",
        }
        (directory / "review-manifest.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        directory.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="Anchored completed Actions result.json")
    parser.add_argument("--experiment", choices=sorted(ALLOWLIST))
    parser.add_argument(
        "--batch-id",
        choices=sorted(ALLOWED_BATCHES),
        help="Defaults to the matching batch in the anchored Actions result",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/modal-course/capstone-review"))
    args = parser.parse_args()
    try:
        target = download_review(args.result, args.output, experiment=args.experiment, batch=args.batch_id)
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        if isinstance(error, ReviewDownloadError):
            parser.error(str(error))
        parser.error("Cannot verify the supplied result or review files; no review artifact was installed")
    print(json.dumps({"review_directory": str(target), "public_release_approved": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
