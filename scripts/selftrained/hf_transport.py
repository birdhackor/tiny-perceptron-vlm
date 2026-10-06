"""Pinned data transport and an explicit public inference-file allowlist.

This module never downloads model weights. Training checkpoints and optimizer
state stay on the existing private course Volume.
"""

import hashlib
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

MAX_PACKAGE_BYTES = 512 * 1024 * 1024
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024
MAX_RELEASE_BYTES = 128 * 1024 * 1024
PUBLIC_MODEL_REPO = "birdhackor/tiny-perceptron-course-models"


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
