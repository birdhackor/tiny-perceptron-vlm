"""Prepare reviewed capstone exports and verify actual public, pinned HF bytes.

Preparing an approval leaves approved/reviewed false. Publishing is performed by
the repository's release process, not by this script. The public manifest is only
written after unauthenticated downloads at the specified HF commit match exports.
"""

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.course_release import (  # noqa: E402
    build_export,
    file_sha256,
    safe_relative,
    validate_export,
    write_model_card,
)
from tiny_perceptron.capstone import DATA_VERSION, STAGES, TOK, CapstoneModel  # noqa: E402
from tiny_perceptron.capstone_quantization import FORMAT as PTQ_FORMAT  # noqa: E402
from tiny_perceptron.capstone_quantization import restore_quantized_payload  # noqa: E402
from tiny_perceptron.model import ModelConfig  # noqa: E402

PUBLIC_REPO = "birdhackor/tiny-perceptron-course-models"
MANIFEST = ROOT / "docs/course-experiments/capstone-public.json"
STAGE_IDS = (*STAGES, "dpo-int4", "dpo-int8")
FORMATS = ("capstone-v1", PTQ_FORMAT)
PROVENANCE_KEYS = {
    "revision",
    "source_revision",
    "source_repo",
    "source_prefix",
    "source_sha256",
    "source_checkpoint_sha256",
    "experiment_id",
    "experiment",
    "batch_id",
    "run_id",
    "training_revision",
    "private_revision",
    "private_repo",
    "private_prefix",
    "source_checkpoint",
    "dataset_manifest_sha256",
    "data_version",
    "seed",
    "approval_git_revision",
    "approval_sha256",
    "scope",
}


def _pinned(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{label} must be a pinned hexadecimal value of length {length}")
    return value


def _metadata(saved, provenance):
    clean = {}
    for source in (saved.get("metadata", {}), provenance):
        if not isinstance(source, dict):
            raise ValueError("Capstone provenance must be an object")
        for key, value in source.items():
            if key in PROVENANCE_KEYS:
                if not isinstance(value, (str, int)) or isinstance(value, bool):
                    raise ValueError("Capstone provenance accepts only explicit text/integer fields")
                clean[key] = value
    return clean


def clean_capstone_payload(saved, provenance, specification):
    """An explicit architecture declaration and strict inference-field allowlist."""
    if specification.get("architecture") != {"type": "CapstoneModel"}:
        raise ValueError("Capstone export requires explicit architecture.type=CapstoneModel")
    if (
        saved.get("format_version") not in FORMATS
        or saved.get("data_version") != DATA_VERSION
        or saved.get("stage") not in STAGES
        or type(saved.get("step")) is not int
        or saved["step"] < 0
        or saved.get("tokenizer") != TOK.state()
    ):
        raise ValueError("Invalid capstone format, stage or byte-token protocol")
    config = saved.get("config", {})
    if not isinstance(config, dict) or set(config) != set(asdict(ModelConfig())):
        raise ValueError("Capstone config must contain exactly the supported ModelConfig fields")
    if config.get("vocab_size") != TOK.vocab_size or config.get("tied") is not False:
        raise ValueError("Capstone requires its byte vocabulary and untied output weights")
    fields = ("format_version", "config", "model", "stage", "step", "tokenizer", "data_version")
    clean = {key: saved[key] for key in fields}
    if saved["format_version"] == PTQ_FORMAT:
        clean.update(quantized=saved["quantized"], quantization=saved["quantization"])
    clean.update(inference_only=True, architecture={"type": "CapstoneModel"}, metadata=_metadata(saved, provenance))
    # Validate state keys, dtypes and shapes before any serialized public output.
    validate_capstone_payload(clean)
    # A tensor view can otherwise serialize its larger, unrelated backing storage.
    clean["config"] = dict(clean["config"])
    clean["tokenizer"] = TOK.state()
    clean["model"] = {name: value.detach().cpu().contiguous().clone() for name, value in clean["model"].items()}
    if "quantized" in clean:
        clean["quantized"] = {
            name: {
                "values": record["values"].detach().cpu().contiguous().clone(),
                "scale": record["scale"].detach().cpu().contiguous().clone(),
                "shape": list(record["shape"]),
            }
            for name, record in clean["quantized"].items()
        }
        clean["quantization"] = dict(
            clean["quantization"], linear_weights=list(clean["quantization"]["linear_weights"])
        )
    return clean


def validate_capstone_payload(saved):
    fields = {
        "format_version",
        "inference_only",
        "config",
        "model",
        "stage",
        "step",
        "tokenizer",
        "data_version",
        "architecture",
        "metadata",
    }
    if saved.get("format_version") == PTQ_FORMAT:
        fields.update(("quantized", "quantization"))
    if set(saved) != fields or saved.get("architecture") != {"type": "CapstoneModel"}:
        raise ValueError("Capstone public inference fields and explicit architecture must match the format")
    config = saved.get("config", {})
    if not isinstance(config, dict) or set(config) != set(asdict(ModelConfig())):
        raise ValueError("Capstone config must contain exactly the supported ModelConfig fields")
    integer_fields = ("vocab_size", "width", "layers", "heads", "max_length", "experts", "top_k")
    if (
        any(type(config[key]) is not int for key in integer_fields)
        or (config["kv_heads"] is not None and type(config["kv_heads"]) is not int)
        or type(config["rotary"]) is not bool
        or config["tied"] is not False
        or config["vocab_size"] != TOK.vocab_size
        or config["norm"] not in ("layer", "rms")
        or config["activation"] not in ("gelu", "swiglu")
        or config["backend"] not in ("manual", "sdpa")
    ):
        raise ValueError("Unsupported capstone configuration value or type")
    if _metadata(saved, {}) != saved.get("metadata"):
        raise ValueError("Capstone public metadata contains unapproved fields")
    if saved.get("format_version") == PTQ_FORMAT:
        model = restore_quantized_payload(saved)
    elif saved.get("format_version") == "capstone-v1":
        if (
            saved.get("inference_only") is not True
            or saved.get("data_version") != DATA_VERSION
            or saved.get("tokenizer") != TOK.state()
            or saved.get("stage") not in STAGES
            or type(saved.get("step")) is not int
            or saved["step"] < 0
        ):
            raise ValueError("Invalid capstone inference checkpoint")
        model = CapstoneModel(ModelConfig(**saved["config"]))
        expected = model.state_dict()
        state = saved["model"]
        if set(state) != set(expected):
            raise ValueError("Capstone tensors must exactly match the declared architecture")
        for name, tensor in state.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.shape != expected[name].shape
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("Capstone weights must be finite float32 tensors of the declared shapes")
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError("Unsupported capstone inference format")
    model.eval()
    with torch.no_grad():
        # Exercise language, image and audio reconstruction, including modalities.
        ids = torch.tensor([[TOK.bos_id, TOK.user_id, TOK.image_id, TOK.audio_id, 8]])
        logits = model(ids, images=torch.zeros(1, 3, 16, 16), audio_features=torch.zeros(1, 16))["logits"]
    if logits.shape != (1, 5, TOK.vocab_size) or not torch.isfinite(logits).all():
        raise ValueError("Capstone CPU forward failed to produce finite byte-vocabulary logits")
    return saved["format_version"]


def prepare_approval(
    directory,
    stage_files,
    *,
    training_revision,
    private_repo,
    private_revision,
    private_prefix,
    batch_id="capstone-v1",
    experiment_id="capstone",
    model_card=None,
):
    """Build an unapproved review draft from actual files; never fabricate a pass."""
    _pinned(training_revision, 40, "Training Git revision")
    _pinned(private_revision, 40, "Private HF revision")
    safe_relative(private_prefix)
    if (
        not isinstance(experiment_id, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", experiment_id)
        or not isinstance(batch_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", batch_id)
        or safe_relative(private_prefix).parts[:3] != ("course", batch_id, experiment_id)
    ):
        raise ValueError("Private prefix must belong to the explicit experiment and batch")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", private_repo):
        raise ValueError("Private repository must be owner/name")
    if not stage_files or set(stage_files) - set(STAGE_IDS):
        raise ValueError("Explicit capstone stage files are required")
    directory = Path(directory).resolve()
    files = []
    for stage_id, relative in stage_files.items():
        path = directory / safe_relative(relative)
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or not path.is_file():
            raise ValueError("Stage checkpoint must be a regular file under the reviewed source directory")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        specification = {"architecture": {"type": "CapstoneModel"}}
        clean = clean_capstone_payload(saved, {}, specification)
        bits = clean.get("quantization", {}).get("bits")
        expected_id = f"{clean['stage']}-int{bits}" if bits else clean["stage"]
        if stage_id != expected_id:
            raise ValueError("Stage label disagrees with actual checkpoint stage/precision")
        files.append(
            {
                "path": relative,
                "output": f"{stage_id}/model.pt",
                "sha256": file_sha256(path),
                "kind": "checkpoint",
                "license": "MIT",
                "redistribution_approved": False,
                "architecture": {"type": "CapstoneModel"},
            }
        )
    return {
        "schema_version": 1,
        "approved": False,
        "reviewed": False,
        "experiment_id": experiment_id,
        "batch_id": batch_id,
        "revision": training_revision,
        "private_source": {"repo": private_repo, "revision": private_revision, "prefix": private_prefix},
        "files": files,
        "model_card": model_card
        or {
            "summary": "An integrated small-world teaching model with one text/multimodal/tool language core.",
            "scope": "Synthetic RGB shapes, low/high tones, narrow Chinese prompts and an external calculator.",
            "limitations": [
                "Not a general assistant; template generalization and every stage require measured evaluation."
            ],
            "training_data": [
                {
                    "source": "capstone-small-world-v1 synthetic data",
                    "license": "MIT",
                    "modifications": "Generated by this MIT-licensed repository.",
                }
            ],
        },
    }


def export_capstone(directory, destination, approval, provenance):
    files = build_export(directory, destination, approval, provenance)
    write_model_card(
        Path(destination) / "README.md",
        approval,
        PUBLIC_REPO,
        f"course/{approval['batch_id']}/{approval['experiment_id']}",
    )
    return files


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("repo") != PUBLIC_REPO:
        raise ValueError("Unsupported capstone public manifest or repository")
    _pinned(manifest.get("revision"), 40, "Public HF revision")
    models = manifest.get("models", [])
    if not models or len({item.get("id") for item in models}) != len(models):
        raise ValueError("Capstone stages must be nonempty and unique")
    for model in models:
        if model.get("id") not in STAGE_IDS or model.get("format_version") not in FORMATS:
            raise ValueError("Unsupported capstone stage or inference format")
        files = model.get("files", [])
        outputs, paths = set(), set()
        if not files:
            raise ValueError("Every capstone stage requires verified files")
        checkpoint = safe_relative(model.get("checkpoint", "")).as_posix()
        for item in files:
            path, output = safe_relative(item["path"]).as_posix(), safe_relative(item["output"]).as_posix()
            if output in outputs or path in paths:
                raise ValueError("Capstone file paths and outputs must be unique per stage")
            outputs.add(output)
            paths.add(path)
            _pinned(item.get("sha256"), 64, "File SHA-256")
            if type(item.get("bytes")) is not int or item["bytes"] < 1 or item.get("license") != "MIT":
                raise ValueError("Capstone files require positive byte counts and their reviewed MIT license")
            if Path(output).suffix in (".pt", ".pth", ".ckpt", ".safetensors") and output != checkpoint:
                raise ValueError("Only the selected inference checkpoint can be downloaded for a stage")
        if checkpoint not in outputs:
            raise ValueError("Capstone checkpoint must be among the verified downloaded files")
    return manifest


def build_public_manifest(export_directory, *, revision, prefix, repo=PUBLIC_REPO):
    """Return a public manifest only after actual anonymous, pinned HF downloads."""
    from huggingface_hub import hf_hub_download

    if repo != PUBLIC_REPO:
        raise ValueError("Capstone publication uses the existing course model repository")
    _pinned(revision, 40, "Public HF revision")
    prefix = safe_relative(prefix).as_posix()
    directory = Path(export_directory)
    exported = json.loads((directory / "export-manifest.json").read_text(encoding="utf-8"))
    common = []
    for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Public model cards and license notices must accompany checkpoints")
        common.append({"output": name, "sha256": file_sha256(path), "bytes": path.stat().st_size, "license": "MIT"})
    models = []
    verified = set()
    for item in exported["files"]:
        output = safe_relative(item["output"]).as_posix()
        if Path(output).suffix != ".pt":
            continue
        checkpoint = directory / output
        if (
            checkpoint.is_symlink()
            or file_sha256(checkpoint) != item["sha256"]
            or checkpoint.stat().st_size != item["bytes"]
        ):
            raise ValueError("Exported checkpoint changed before publication verification")
        validate_export(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        bits = saved.get("quantization", {}).get("bits")
        stage_id = f"{saved['stage']}-int{bits}" if bits else saved["stage"]
        files = [
            {
                "output": "model.pt",
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "license": item["license"],
                "path": f"{prefix}/{output}",
            }
        ]
        files += [dict(entry, path=f"{prefix}/{entry['output']}") for entry in common]
        for entry in files:
            if entry["path"] in verified:
                continue
            cached = Path(hf_hub_download(repo, entry["path"], revision=revision, token=False))
            if file_sha256(cached) != entry["sha256"] or cached.stat().st_size != entry["bytes"]:
                raise ValueError("Actual public HF bytes differ from the reviewed export")
            verified.add(entry["path"])
        models.append(
            {"id": stage_id, "format_version": saved["format_version"], "checkpoint": "model.pt", "files": files}
        )
    return validate_manifest({"schema_version": 1, "repo": repo, "revision": revision, "models": models})


def build_public_manifest_from_receipt(public_manifest):
    """Verify the final Modal public receipt, including its rewritten pinned card.

    Modal publishes weights first and rewrites README in a second HF commit. A
    pre-upload local README therefore cannot represent the final public bytes.
    This entrypoint checks both receipt hashes and the downloaded export manifest.
    """
    from huggingface_hub import hf_hub_download

    if public_manifest.get("repo") != PUBLIC_REPO:
        raise ValueError("Capstone publication uses the existing course model repository")
    revision = _pinned(public_manifest.get("revision"), 40, "Public HF revision")
    files = public_manifest.get("files", [])
    if not files:
        raise ValueError("A real public release receipt with files is required")
    declared = {}
    outputs = set()
    for item in files:
        path = safe_relative(item["path"]).as_posix()
        output = safe_relative(item["output"]).as_posix()
        _pinned(item.get("sha256"), 64, "Receipt file SHA-256")
        if path in declared or output in outputs or type(item.get("bytes")) is not int or item["bytes"] < 1:
            raise ValueError("Public receipt requires unique files with exact positive byte counts")
        declared[path] = item
        outputs.add(output)
    by_output = {item["output"]: item for item in files}
    if any(
        name not in by_output for name in ("export-manifest.json", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md")
    ):
        raise ValueError("Public receipt lacks export provenance, model card or license notices")

    def verified(item):
        cached = Path(hf_hub_download(PUBLIC_REPO, item["path"], revision=revision, token=False))
        if file_sha256(cached) != item["sha256"] or cached.stat().st_size != item["bytes"]:
            raise ValueError("Actual public HF bytes differ from the completed release receipt")
        return cached

    exported = json.loads(verified(by_output["export-manifest.json"]).read_text(encoding="utf-8"))
    common = []
    for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        item = by_output[name]
        verified(item)
        common.append({key: item[key] for key in ("path", "output", "sha256", "bytes")} | {"license": "MIT"})
    models = []
    for item in exported["files"]:
        output = safe_relative(item["output"]).as_posix()
        if Path(output).suffix != ".pt":
            continue
        receipt = by_output.get(output)
        if (
            not receipt
            or receipt["sha256"] != item["sha256"]
            or receipt["bytes"] != item["bytes"]
            or item.get("license") != "MIT"
        ):
            raise ValueError("Published checkpoint disagrees with its reviewed export manifest")
        checkpoint = verified(receipt)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validate_capstone_payload(saved)
        bits = saved.get("quantization", {}).get("bits")
        stage_id = f"{saved['stage']}-int{bits}" if bits else saved["stage"]
        download = {key: receipt[key] for key in ("path", "sha256", "bytes")}
        download.update(output="model.pt", license="MIT")
        models.append(
            {
                "id": stage_id,
                "format_version": saved["format_version"],
                "checkpoint": "model.pt",
                "files": [download, *common],
            }
        )
    return validate_manifest({"schema_version": 1, "repo": PUBLIC_REPO, "revision": revision, "models": models})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument(
        "--stages", type=Path, required=True, help="JSON object mapping stage IDs to actual source checkpoint paths"
    )
    prepare.add_argument("--training-revision", required=True)
    prepare.add_argument("--private-repo", required=True)
    prepare.add_argument("--private-revision", required=True)
    prepare.add_argument("--private-prefix", required=True)
    prepare.add_argument("--batch-id", default="capstone-v1")
    prepare.add_argument("--experiment-id", default="capstone")
    prepare.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--approval", type=Path, required=True)
    export.add_argument("--provenance", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    manifest = sub.add_parser("verify-public")
    manifest.add_argument("--export", type=Path, required=True)
    manifest.add_argument("--revision", required=True)
    manifest.add_argument("--prefix", required=True)
    manifest.add_argument("--output", type=Path, default=MANIFEST)
    receipt = sub.add_parser("verify-receipt")
    receipt.add_argument(
        "--receipt", type=Path, required=True, help="JSON object containing the completed Modal public_manifest"
    )
    receipt.add_argument("--output", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_approval(
            args.source,
            json.loads(args.stages.read_text()),
            training_revision=args.training_revision,
            private_repo=args.private_repo,
            private_revision=args.private_revision,
            private_prefix=args.private_prefix,
            batch_id=args.batch_id,
            experiment_id=args.experiment_id,
        )
    elif args.command == "export":
        result = export_capstone(
            args.source, args.output, json.loads(args.approval.read_text()), json.loads(args.provenance.read_text())
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    elif args.command == "verify-public":
        result = build_public_manifest(args.export, revision=args.revision, prefix=args.prefix)
    else:
        value = json.loads(args.receipt.read_text(encoding="utf-8"))
        result = build_public_manifest_from_receipt(value.get("public_manifest", value))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(args.output))


if __name__ == "__main__":
    main()
