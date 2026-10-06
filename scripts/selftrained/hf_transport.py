"""Pinned data transport and an explicit public inference-file allowlist.

This module never supplies external weights to training. Release recovery reads
only previously reviewed public files; optimizer state stays on the private Volume.
"""

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024
MAX_RELEASE_BYTES = 128 * 1024 * 1024
PUBLIC_MODEL_REPO = "birdhackor/tiny-perceptron-course-models"
REPOSITORY_CARD_SOURCE = "docs/selftrained/model-cards/repository-README.md"
MAX_REPOSITORY_CARD_BYTES = 64 * 1024
EXPORT_STAGES = {
    "moe-pretrain": ("moe", "pretrain"),
    "moe-sft": ("moe", "sft"),
    "moe-joint": ("moe", "joint"),
    "dense-joint": ("dense", "joint"),
}
SAFE_EXPORT_FILES = {"model.safetensors", "model-config.json", "tokenizer.json", "inference-manifest.json"}
PUBLIC_EVALUATION_PREFIXES = {
    "selftrained/v2/moe-joint": "moe",
    "selftrained/v2/dense-joint": "dense",
}
# The V2 export contract is fixed; control images validate it with stdlib only.
PUBLIC_EVALUATION_PREPROCESS = "gray-crops32-ocr-letterbox-full-logmel40-v2"
PUBLIC_TOKENIZER_SPECIALS = [
    f"<{name}>" for name in ("pad", "bos", "eos", "user", "assistant", "system", "tool", "image", "ocr", "audio", "unk")
]


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate public inference metadata key")
        result[key] = value
    return result


def validate_public_evaluation(value, architecture):
    """A public evaluation may recover only four reviewed V2 inference files."""
    if architecture not in ("moe", "dense"):
        raise ValueError("Public evaluation architecture must be moe or dense")
    if not isinstance(value, dict) or set(value) != {"repo_id", "revision", "prefix", "files"}:
        raise ValueError("Public evaluation needs the exact immutable four-file descriptor")
    if value["repo_id"] != PUBLIC_MODEL_REPO:
        raise ValueError("Public evaluation must use the authorized model repository")
    if not isinstance(value["revision"], str) or not re.fullmatch(r"[a-f0-9]{40}", value["revision"]):
        raise ValueError("Public evaluation requires an immutable full HF commit")
    if not isinstance(value["prefix"], str) or PUBLIC_EVALUATION_PREFIXES.get(value["prefix"]) != architecture:
        raise ValueError("Public evaluation prefix must be the authorized V2 architecture's joint export")
    files = value["files"]
    if not isinstance(files, list) or len(files) != 4:
        raise ValueError("Public evaluation requires exactly four safe inference files")
    seen, total = set(), 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError("Public evaluation file needs only its fixed path, SHA and bytes")
        if not isinstance(item["path"], str) or item["path"] not in SAFE_EXPORT_FILES or item["path"] in seen:
            raise ValueError("Public evaluation may download only the unique four safe files")
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise ValueError("Each public inference file needs exact SHA-256")
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise ValueError("Public inference files need positive bounded byte counts")
        total += item["bytes"]
        seen.add(item["path"])
    if total > MAX_RELEASE_BYTES:
        raise ValueError("Public evaluation exceeds the 128 MiB download allowance")
    return {item["path"]: item for item in files}


def public_evaluation_identity(root, public, checkpoint, manifest, architecture, training=None):
    """Validate metadata against the frozen data/config and trusted private lineage.

    This helper needs no torch or checkpoint deserialization. The caller verifies
    payload bytes separately; control containers read only three small JSON files.
    """
    files = validate_public_evaluation(public, architecture)
    if checkpoint.get("stage") != "joint" or checkpoint.get("path") != "best.pt":
        raise ValueError("Public evaluation requires a selected private joint checkpoint lineage")
    inference = json.loads(
        verify_file(root, files["inference-manifest.json"]).read_text(), object_pairs_hook=unique_json_object
    )
    config = json.loads(verify_file(root, files["model-config.json"]).read_text(), object_pairs_hook=unique_json_object)
    if not isinstance(inference, dict) or not isinstance(config, dict):
        raise ValueError("Public inference metadata/config must be JSON objects")
    tokenizer = json.loads(verify_file(root, files["tokenizer.json"]).read_text(), object_pairs_hook=unique_json_object)
    if not isinstance(tokenizer, dict) or set(tokenizer) != {"type", "specials", "characters"}:
        raise ValueError("Public tokenizer must have the exact character vocabulary schema")
    characters = tokenizer["characters"]
    if (
        tokenizer["type"] != "selftrained_char_v1"
        or tokenizer["specials"] != PUBLIC_TOKENIZER_SPECIALS
        or not isinstance(characters, list)
        or any(not isinstance(char, str) or len(char) != 1 for char in characters)
        or characters != sorted(set(characters))
        or type(config.get("vocab_size")) is not int
        or config["vocab_size"] != len(PUBLIC_TOKENIZER_SPECIALS) + len(characters)
    ):
        raise ValueError("Public tokenizer vocabulary/role IDs differ from its exact model config")
    tokenizer_sha = hashlib.sha256(json.dumps(tokenizer, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    records = {Path(item["path"]).name: item["sha256"] for item in manifest["records"]}
    assets = {item["path"]: item["sha256"] for item in manifest.get("assets", [])}
    asset_hashes = inference.get("asset_sha256", {})
    if (
        inference.get("schema") != "selftrained-random-v1"
        or inference.get("preprocess_version") != PUBLIC_EVALUATION_PREPROCESS
        or inference.get("selected_checkpoint_sha256") != checkpoint["sha256"]
        or inference.get("stage") != "joint"
        or inference.get("origin", {}).get("kind") != "all-neural-weights-random"
        or inference.get("selection") != "validation_loss"
        or inference.get("files")
        != {name: item["sha256"] for name, item in files.items() if name != "inference-manifest.json"}
        or inference.get("data_sha256") != records
        or not isinstance(asset_hashes, dict)
        or not asset_hashes
        or any(assets.get(path) != value for path, value in asset_hashes.items())
        or inference.get("tokenizer_sha256") != tokenizer_sha
        or config.get("architecture") != architecture
        or any(
            config.get(key) != value
            for key, value in manifest["model_config"].items()
            if key not in ("vocab_size", "architecture")
        )
    ):
        raise ValueError("Public safe export differs from the selected random checkpoint or frozen data/config")
    if training is not None and (
        training.get("config") != config
        # Current genuine receipts omit this field; never require a fabricated
        # value or claim a direct comparison with absent training metadata.
        or training.get("preprocess_version", PUBLIC_EVALUATION_PREPROCESS) != PUBLIC_EVALUATION_PREPROCESS
        or training.get("data_sha256") != records
        or training.get("asset_sha256") != asset_hashes
        or training.get("tokenizer_sha256") != inference["tokenizer_sha256"]
        or training.get("architecture") != architecture
        or training.get("stage") != "joint"
        or training.get("origin", {}).get("kind") != "all-neural-weights-random"
        or training.get("completed_requested_steps") is not True
        or training.get("selected_checkpoint_available") is not True
        or training.get("inference_exported") is not True
        or training.get("test_used_for_selection") is not False
    ):
        raise ValueError("Public safe export differs from the completed trusted training receipt")
    return {
        "weight_source": "safe_export",
        "checkpoint_sha256": files["model.safetensors"]["sha256"],
        "safe_weights_sha256": files["model.safetensors"]["sha256"],
        "inference_manifest_sha256": files["inference-manifest.json"]["sha256"],
        "selected_checkpoint_sha256": checkpoint["sha256"],
    }


def download_public_evaluation(public, destination, checkpoint, manifest, architecture):
    """Fetch only pinned public files with authentication explicitly disabled."""
    files = validate_public_evaluation(public, architecture)
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Public safe export destination may not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    if any(path.name not in SAFE_EXPORT_FILES or path.is_symlink() for path in destination.iterdir()):
        raise ValueError("Public safe export destination contains unexpected files or links")
    # Check immutable commit and advertised byte count before any payload fetch.
    for name, item in files.items():
        url = hf_hub_url(
            public["repo_id"],
            filename=f"{public['prefix']}/{name}",
            revision=public["revision"],
            repo_type="model",
            endpoint="https://huggingface.co",
        )
        metadata = get_hf_file_metadata(url, token=False)
        if metadata.size != item["bytes"] or metadata.commit_hash != public["revision"]:
            raise ValueError("Public HF file metadata differs from the pinned commit/bytes")
    for name, item in files.items():
        target = destination / name
        if target.exists():
            verify_file(destination, item)
            continue
        source = Path(
            hf_hub_download(
                repo_id=public["repo_id"],
                filename=f"{public['prefix']}/{name}",
                revision=public["revision"],
                repo_type="model",
                token=False,
                endpoint="https://huggingface.co",
            )
        )
        if not source.is_file() or source.stat().st_size != item["bytes"] or digest(source) != item["sha256"]:
            raise ValueError("Actual public HF file differs from pinned bytes/SHA")
        shutil.copyfile(source, target)
        verify_file(destination, item)
    identity = public_evaluation_identity(destination, public, checkpoint, manifest, architecture)
    return {
        "public_export": public,
        "authentication": "disabled",
        "weight_source": "safe_export",
        "files": list(files.values()),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        **identity,
    }


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative_path(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Need a relative path without traversal")
    return path


def verify_file(root, item):
    path = Path(root) / relative_path(item["path"])
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Manifest file is absent or outside the package")
    if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
        raise ValueError(f"Size/hash mismatch: {item['path']}")
    return path


def unpack_verified_archive(archive, destination, package):
    archive, destination = Path(archive), Path(destination)
    if archive.stat().st_size != package["bytes"] or digest(archive) != package["sha256"]:
        raise ValueError("Downloaded dataset archive differs from pinned bytes")
    if not 0 < package["bytes"] <= MAX_PACKAGE_BYTES:
        raise ValueError("Dataset archive exceeds transport byte cap")
    if not 0 < package["unpacked_bytes"] <= MAX_UNPACKED_BYTES:
        raise ValueError("Dataset package exceeds extracted byte cap")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:
        members, total, paths = [], 0, set()
        for member in source:
            if len(members) >= 20000:
                raise ValueError("Archive exceeds bounded member count")
            path = relative_path(member.name)
            if path.as_posix() in paths or not (member.isfile() or member.isdir()):
                raise ValueError("Archive contains duplicate, link or special member")
            paths.add(path.as_posix())
            total += member.size
            if total > package["unpacked_bytes"]:
                raise ValueError("Archive expands beyond its pinned byte allowance")
            members.append(member)
        if total != package["unpacked_bytes"]:
            raise ValueError("Archive extracted size differs from pinned byte total")
        source.extractall(destination, members=members, filter="data")
    return destination


def materialize_git_lfs(package, repository, destination):
    """Fetch just the pinned package using the repository's existing Git auth."""
    repository, destination = Path(repository), Path(destination)
    if package.get("kind") != "git-lfs" or package.get("repo_id") != "birdhackor/tiny-perceptron-vlm":
        raise ValueError("Main-line data must use the existing Git LFS repository")
    if not re.fullmatch(r"[a-f0-9]{40}", package["revision"]):
        raise ValueError("Git LFS package needs exact full commit")
    path = relative_path(package["path"])
    if path.parts[:2] != ("assets", "training") or path.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("Package must be a committed tar.gz under assets/training")
    pointer = subprocess.run(
        ["git", "show", f"{package['revision']}:{path.as_posix()}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    expected = f"version https://git-lfs.github.com/spec/v1\noid sha256:{package['sha256']}\nsize {package['bytes']}\n"
    if pointer != expected:
        raise ValueError("Committed Git LFS pointer differs from transport manifest")
    subprocess.run(
        ["git", "lfs", "fetch", "origin", package["revision"], f"--include={path.as_posix()}", "--exclude="],
        cwd=repository,
        check=True,
    )
    git_dir = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], cwd=repository, capture_output=True, text=True, check=True
    ).stdout.strip()
    oid = package["sha256"]
    source = repository / git_dir / "lfs/objects" / oid[:2] / oid[2:4] / oid
    if not source.is_file() or source.stat().st_size != package["bytes"] or digest(source) != oid:
        raise ValueError("Actual fetched Git LFS object differs from pinned size/hash")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def approved_public_files(source, release, revision, manifest_sha):
    if release.get("repo_id") != PUBLIC_MODEL_REPO or release.get("private") is not False:
        raise ValueError("Public release must name the authorized model repository")
    if release.get("revision") != revision or release.get("manifest_sha256") != manifest_sha:
        raise ValueError("Public release must bind exact source Git SHA and data manifest")
    prefix = relative_path(release.get("prefix", ""))
    if prefix.parts[0] != "selftrained" or len(prefix.parts) < 2:
        raise ValueError("Release prefix must be versioned under selftrained/")
    files = release.get("files", [])
    if not files or len(files) > 20:
        raise ValueError("Release requires a nonempty bounded file allowlist")
    total, seen = 0, set()
    for item in files:
        path = relative_path(item["path"])
        if path.as_posix() in seen or item.get("kind") not in ("inference", "metadata", "model_card", "license"):
            raise ValueError("Release files need unique names and reviewed public kinds")
        if item.get("redistribution_approved") is not True or not item.get("license"):
            raise ValueError("Every public file needs explicit redistribution and license metadata")
        if path.suffix not in (".safetensors", ".json", ".md", ".txt"):
            raise ValueError("Resume checkpoints, raw data and unknown files remain private")
        verify_file(source, item)
        total += item["bytes"]
        seen.add(path.as_posix())
    if total > MAX_RELEASE_BYTES:
        raise ValueError("Public inference export exceeds bounded release allowance")
    return files


def publish_inference(source, release, revision, manifest_sha, token, staging):
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import EntryNotFoundError

    files = approved_public_files(source, release, revision, manifest_sha)
    api = HfApi(token=token)
    info = api.repo_info(PUBLIC_MODEL_REPO, repo_type="model")
    if info.private:
        raise ValueError("Authorized model repository is expected to be public")
    try:
        existing = next(
            iter(
                api.list_repo_tree(
                    PUBLIC_MODEL_REPO, repo_type="model", revision=info.sha, path_in_repo=release["prefix"]
                )
            )
        )
    except (EntryNotFoundError, StopIteration):
        existing = None
    if existing is not None:
        raise ValueError("Release prefix already exists; never overwrite a prior public release")
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    operations = []
    for item in files:
        relative = relative_path(item["path"])
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / relative, target)
        verify_file(staging, item)
        operations.append(
            CommitOperationAdd(path_in_repo=f"{release['prefix']}/{relative.as_posix()}", path_or_fileobj=target)
        )
    result = api.create_commit(
        repo_id=PUBLIC_MODEL_REPO,
        repo_type="model",
        operations=operations,
        commit_message=f"Self-trained inference release {revision[:12]}",
        parent_commit=info.sha,
    )
    return {
        "repo_id": PUBLIC_MODEL_REPO,
        "repo_type": "model",
        "private": False,
        "commit_sha": result.oid,
        "commit_url": result.commit_url,
        "prefix": release["prefix"],
        "files": files,
        "scope": "Only reviewed inference files; full training/resume state retained privately on course Volume",
    }


def validate_batch_release(release):
    if release.get("repo_id") != PUBLIC_MODEL_REPO or release.get("private") is not False:
        raise ValueError("Batch release must name the authorized public model repository")
    prefix = relative_path(release.get("prefix", ""))
    if prefix.parts[0] != "selftrained" or len(prefix.parts) < 2:
        raise ValueError("Batch release prefix must be versioned under selftrained/")
    if not re.fullmatch(r"[a-f0-9]{64}", release.get("manifest_sha256", "")):
        raise ValueError("Batch release needs the exact frozen data/config manifest")
    exports = release.get("exports", [])
    if len(exports) != 4 or {item.get("name") for item in exports} != set(EXPORT_STAGES):
        raise ValueError("Batch release needs exactly MoE pretrain/SFT/joint and Dense joint")
    for item in exports:
        architecture, stage = EXPORT_STAGES[item["name"]]
        source = item.get("source", {})
        if item.get("architecture") != architecture or source.get("stage") != stage or source.get("path") != "best.pt":
            raise ValueError("Public export must name its exact architecture/stage selected checkpoint")
        for key in ("execution_sha256", "train_receipt_sha256"):
            if not re.fullmatch(r"[a-f0-9]{64}", item.get(key, "")):
                raise ValueError("Each export needs exact execution and training receipt hashes")
        if not re.fullmatch(r"[a-f0-9]{40}", item.get("revision", "")):
            raise ValueError("Each export needs its original full source Git revision")
        files = item.get("files", [])
        if len(files) != 4 or {file.get("path") for file in files} != SAFE_EXPORT_FILES:
            raise ValueError("Each export publishes exactly four reviewed safe inference files")
    return exports


def approved_batch_exports(sources, release, manifest_sha, manifest):
    """Verify every private source and safe file before reserving or publishing."""
    exports = validate_batch_release(release)
    if release["manifest_sha256"] != manifest_sha or set(sources) != set(EXPORT_STAGES):
        raise ValueError("Batch release sources differ from the frozen manifest or four-export set")
    expected_records = {Path(item["path"]).name: item["sha256"] for item in manifest["records"]}
    expected_assets = {item["path"]: item["sha256"] for item in manifest.get("assets", [])}
    approved, total = [], 0
    for export in exports:
        source = Path(sources[export["name"]]["directory"])
        execution = sources[export["name"]]["execution"]
        if (
            digest(source / "execution.json") != export["execution_sha256"]
            or execution.get("status") != "completed"
            or execution.get("revision") != export["revision"]
            or execution.get("manifest_sha256") != manifest_sha
            or execution.get("stage") != export["source"]["stage"]
            or execution.get("run_id") != export["source"]["run_id"]
            or execution.get("job", {}).get("architecture", "moe") != export["architecture"]
            or digest(source / "best.pt") != export["source"]["sha256"]
        ):
            raise ValueError("Export source differs from its completed immutable execution/checkpoint")
        receipt_path = source / "train-receipt.json"
        if digest(receipt_path) != export["train_receipt_sha256"]:
            raise ValueError("Export training receipt differs from its reviewed hash")
        receipt = json.loads(receipt_path.read_text())
        asset_hashes = receipt.get("asset_sha256", {})
        if (
            receipt.get("architecture") != export["architecture"]
            or receipt.get("stage") != export["source"]["stage"]
            or receipt.get("data_sha256") != expected_records
            or not asset_hashes
            or any(expected_assets.get(path) != value for path, value in asset_hashes.items())
            or receipt.get("completed_requested_steps") is not True
            or receipt.get("selected_checkpoint_available") is not True
            or receipt.get("inference_exported") is not True
            or receipt.get("test_used_for_selection") is not False
            or receipt.get("origin", {}).get("kind") != "all-neural-weights-random"
            or receipt.get("steps", -1) < execution.get("job", {}).get("steps", 0)
        ):
            raise ValueError(
                "Only completed from-random training with frozen data and validation selection can publish"
            )
        files = approved_public_files(
            source,
            release
            | {
                "revision": export["revision"],
                "prefix": release["prefix"] + "/" + export["name"],
                "files": export["files"],
            },
            export["revision"],
            manifest_sha,
        )
        inference = json.loads((source / "inference-manifest.json").read_text())
        config = json.loads((source / "model-config.json").read_text())
        file_hashes = {item["path"]: item["sha256"] for item in files if item["path"] != "inference-manifest.json"}
        if (
            inference.get("selected_checkpoint_sha256") != export["source"]["sha256"]
            or inference.get("files") != file_hashes
            or inference.get("stage") != export["source"]["stage"]
            or inference.get("selection") != "validation_loss"
            or inference.get("origin", {}).get("kind") != "all-neural-weights-random"
            or config != receipt.get("config")
            or config.get("architecture") != export["architecture"]
        ):
            raise ValueError("Safe inference manifest/config differs from the selected private checkpoint")
        total += sum(item["bytes"] for item in files)
        approved.append({"descriptor": export, "source": source, "files": files})
    if total > MAX_RELEASE_BYTES:
        raise ValueError("The complete four-export batch exceeds the 128 MiB release allowance")
    return approved


def publish_batch_inference(sources, release, manifest_sha, manifest, token, staging):
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    approved = approved_batch_exports(sources, release, manifest_sha, manifest)
    api = HfApi(token=token)
    info = api.repo_info(PUBLIC_MODEL_REPO, repo_type="model")
    if info.private or not re.fullmatch(r"[a-f0-9]{40}", info.sha):
        raise ValueError("Authorized repository must be public with an exact commit revision")
    expected = {
        f"{release['prefix']}/{item['descriptor']['name']}/{file['path']}": file
        for item in approved
        for file in item["files"]
    }
    try:
        existing = {
            item.path: item
            for item in api.list_repo_tree(
                PUBLIC_MODEL_REPO, repo_type="model", revision=info.sha, path_in_repo=release["prefix"], recursive=True
            )
            if hasattr(item, "size")
        }
    except EntryNotFoundError:
        existing = {}
    recovered = bool(existing)
    if existing:
        if set(existing) != set(expected):
            raise ValueError("Immutable public prefix already contains a different file set")
        for path, file in expected.items():
            item = existing[path]
            lfs = getattr(item, "lfs", None)
            lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if item.size != file["bytes"] or (lfs_sha and lfs_sha != file["sha256"]):
                raise ValueError("Existing public release differs from the reviewed safe bytes")
            if not lfs_sha:
                # Only our exact, bounded reviewed files are read for recovery;
                # this path never supplies external weights to training.
                local = Path(
                    hf_hub_download(PUBLIC_MODEL_REPO, path, repo_type="model", revision=info.sha, token=token)
                )
                if local.stat().st_size != file["bytes"] or digest(local) != file["sha256"]:
                    raise ValueError("Existing public release differs from the reviewed safe bytes")
        commit_sha, commit_url = info.sha, f"https://huggingface.co/{PUBLIC_MODEL_REPO}/commit/{info.sha}"
    else:
        staging = Path(staging)
        operations = []
        for item in approved:
            name = item["descriptor"]["name"]
            for file in item["files"]:
                target = staging / name / file["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item["source"] / file["path"], target)
                verify_file(staging / name, file)
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"{release['prefix']}/{name}/{file['path']}", path_or_fileobj=target
                    )
                )
        result = api.create_commit(
            repo_id=PUBLIC_MODEL_REPO,
            repo_type="model",
            operations=operations,
            commit_message="Four reviewed self-trained inference exports",
            parent_commit=info.sha,
        )
        commit_sha, commit_url = result.oid, result.commit_url
    return {
        "repo_id": PUBLIC_MODEL_REPO,
        "repo_type": "model",
        "private": False,
        "commit_sha": commit_sha,
        "commit_url": commit_url,
        "prefix": release["prefix"],
        "manifest_sha256": manifest_sha,
        "exports": release["exports"],
        "files": [
            {"path": path, **{key: value for key, value in file.items() if key != "path"}}
            for path, file in expected.items()
        ],
        "recovered_existing_revision": recovered,
        "new_commit_created": not recovered,
        "scope": "Exactly four safe exports in one immutable public revision; optimizer/resume state stays private",
    }


def validate_repository_card_release(release):
    """One separately approved root card; no arbitrary upload paths or exports."""
    fields = {"mode", "repo_id", "private", "parent_commit", "manifest_sha256", "confirm_write", "file"}
    if not isinstance(release, dict) or set(release) != fields:
        raise ValueError("Repository card requires its exact one-file release descriptor")
    if (
        release["mode"] != "repository-card"
        or release["repo_id"] != PUBLIC_MODEL_REPO
        or release["private"] is not False
        or release["confirm_write"] != "README.md"
    ):
        raise ValueError("Only an explicitly confirmed README.md in the authorized public model repo may change")
    for key, length in (("parent_commit", 40), ("manifest_sha256", 64)):
        if not isinstance(release[key], str) or not re.fullmatch(f"[a-f0-9]{{{length}}}", release[key]):
            raise ValueError("Repository card needs exact HF parent and frozen manifest pins")
    item = release["file"]
    if not isinstance(item, dict) or set(item) != {
        "path",
        "sha256",
        "bytes",
        "kind",
        "license",
        "redistribution_approved",
    }:
        raise ValueError("Repository card requires its exact source file descriptor")
    if (
        item["path"] != REPOSITORY_CARD_SOURCE
        or item["kind"] != "model_card"
        or item["license"] != "MIT"
        or item["redistribution_approved"] is not True
        or not isinstance(item["sha256"], str)
        or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])
        or type(item["bytes"]) is not int
        or not 0 < item["bytes"] <= MAX_REPOSITORY_CARD_BYTES
    ):
        raise ValueError("Repository card must use the fixed reviewed MIT source and bounded bytes/SHA")
    return item


def approved_repository_card(payload, release, manifest_sha):
    item = validate_repository_card_release(release)
    if release["manifest_sha256"] != manifest_sha:
        raise ValueError("Repository card differs from the exact frozen manifest")
    if (
        type(payload) is not bytes
        or len(payload) != item["bytes"]
        or hashlib.sha256(payload).hexdigest() != item["sha256"]
    ):
        raise ValueError("Actual repository card bytes differ from the reviewed Git source")
    # Keep the original bytes for publishing; never serialize/reformat the card.
    text = payload.decode("utf-8")
    if "\x00" in text:
        raise ValueError("Repository card must be UTF-8 text without NUL")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError("Repository card requires canonical YAML front matter")
    header = text[4:].split("\n---\n", 1)[0]
    licenses = re.findall(r"^[ \t]*[\"']?license[\"']?[ \t]*:.*$", header, re.MULTILINE)
    if licenses != ["license: mit"]:
        raise ValueError("Repository card front matter must declare license: mit")
    return item


def repository_card_roundtrip(release, revision):
    """Verify only the immutable public root README with auth disabled."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    item = validate_repository_card_release(release)
    url = hf_hub_url(
        PUBLIC_MODEL_REPO, "README.md", repo_type="model", revision=revision, endpoint="https://huggingface.co"
    )
    metadata = get_hf_file_metadata(url, token=False)
    if metadata.commit_hash != revision or metadata.size != item["bytes"]:
        raise ValueError("Anonymous README metadata differs from the exact commit/bytes")
    local = Path(
        hf_hub_download(
            PUBLIC_MODEL_REPO,
            "README.md",
            repo_type="model",
            revision=revision,
            token=False,
            endpoint="https://huggingface.co",
        )
    )
    if not local.is_file() or local.stat().st_size != item["bytes"] or digest(local) != item["sha256"]:
        raise ValueError("Anonymous README bytes differ from the reviewed card")
    return {
        "path": "README.md",
        "bytes": item["bytes"],
        "sha256": item["sha256"],
        "authentication": "disabled",
        "revision": revision,
    }


def repository_card_public_gate(payload, release):
    from huggingface_hub import HfApi, ModelCard

    item = approved_repository_card(payload, release, release["manifest_sha256"])
    if ModelCard(payload.decode("utf-8")).data.license != "mit":
        raise ValueError("Actual repository card YAML license differs from MIT")
    api = HfApi(token=False, endpoint="https://huggingface.co")
    info = api.repo_info(PUBLIC_MODEL_REPO, repo_type="model", revision="main")
    if info.private is not False or info.sha != release["parent_commit"]:
        raise ValueError("Public repository HEAD differs from the reviewed exact HF parent; do not retry blindly")
    paths = api.get_paths_info(PUBLIC_MODEL_REPO, ["README.md"], repo_type="model", revision=info.sha, token=False)
    if len(paths) > 1 or any(path.path != "README.md" for path in paths):
        raise ValueError("Public README preflight returned unexpected paths")
    blob_sha = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    no_op = bool(paths and paths[0].size == item["bytes"] and paths[0].blob_id == blob_sha)
    public = repository_card_roundtrip(release, info.sha) if no_op else None
    return {"parent_commit": info.sha, "no_op": no_op, "public_verification": public}


def publish_repository_card(payload, release, manifest_sha, token):
    from huggingface_hub import CommitOperationAdd, HfApi

    item = approved_repository_card(payload, release, manifest_sha)
    gate = repository_card_public_gate(payload, release)
    parent = gate["parent_commit"]
    # A single call, no automatic retry or recovery after an ambiguous API failure.
    commit_sha, commit_url = parent, f"https://huggingface.co/{PUBLIC_MODEL_REPO}/commit/{parent}"
    public = gate["public_verification"]
    if not gate["no_op"]:
        operation = CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=payload)
        result = HfApi(token=token, endpoint="https://huggingface.co").create_commit(
            repo_id=PUBLIC_MODEL_REPO,
            repo_type="model",
            revision="main",
            create_pr=False,
            operations=[operation],
            commit_message=f"Reviewed repository card {item['sha256'][:12]}",
            parent_commit=parent,
        )
        # Pinned hub SDK1.33 sets this only after _send_commit; its no-op return
        # bypasses parent CAS. A missing/false flag is ambiguous, never success.
        if getattr(operation, "_is_committed", False) is not True:
            raise ValueError("HF SDK skipped the commit; preserve ambiguous attempt and review the public revision")
        if not isinstance(result.oid, str) or not re.fullmatch(r"[a-f0-9]{40}", result.oid) or result.oid == parent:
            raise ValueError("HF card commit did not return a new immutable full revision")
        commit_sha, commit_url = result.oid, result.commit_url
        public = repository_card_roundtrip(release, commit_sha)
    return {
        "repo_id": PUBLIC_MODEL_REPO,
        "repo_type": "model",
        "private": False,
        "mode": "repository-card",
        "parent_commit": parent,
        "commit_sha": commit_sha,
        "commit_url": commit_url,
        "files": [public],
        "source_file": item,
        "no_op": gate["no_op"],
        "new_commit_created": not gate["no_op"],
        "recovered_existing_revision": False,
        "scope": "Exactly one root README.md Add; no inference export or other repository path operation",
    }
