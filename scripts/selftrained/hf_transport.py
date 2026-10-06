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
EXPORT_STAGES = {
    "moe-pretrain": ("moe", "pretrain"),
    "moe-sft": ("moe", "sft"),
    "moe-joint": ("moe", "joint"),
    "dense-joint": ("dense", "joint"),
}
SAFE_EXPORT_FILES = {"model.safetensors", "model-config.json", "tokenizer.json", "inference-manifest.json"}


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
