#!/usr/bin/env python3
"""Verify and reproducibly package the frozen selftrained data recipe.

This CLI is standard-library-only and never uploads anything. Reconstruction
executes only the three checksum-pinned, committed project data producers. A
local first freeze uses already existing files; it cannot overwrite a freeze.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zlib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
MAX_FILES = 20_000
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024
PRODUCERS = (
    "scripts/selftrained/prepare_vision_ocr.py",
    "scripts/selftrained/prepare_voice.py",
    "scripts/selftrained/prepare_text_tools.py",
)
BLOCKED_PRODUCER_ARGS = frozenset(
    (
        "--data-dir",
        "--data-root",
        "--output-dir",
        "--manifest-dir",
        "--voice-audit",
        "--research-supplements",
        "--verify-only",
    )
)
ALLOWED_PRODUCER_FLAGS = {
    PRODUCERS[0]: frozenset(
        ("--train-per-class", "--validation-per-class", "--test-per-class", "--omit-heldout-single-characters")
    ),
    PRODUCERS[1]: frozenset(("--offline",)),
    PRODUCERS[2]: frozenset(),
}


class PackageError(ValueError):
    """The recipe, input files, or resulting archive violate the contract."""


def safe_relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise PackageError("paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or path == PurePosixPath("."):
        raise PackageError(f"noncanonical or traversing relative path: {value!r}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def byte_count(value: object, description: str, *, limit: int | None = None) -> int:
    if type(value) is not int or value < 0 or (limit is not None and value > limit):
        raise PackageError(f"invalid or excessive {description}")
    return value


def regular_source(root: Path, relative: str) -> Path:
    path = root
    for component in safe_relative(relative).parts:
        path /= component
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise PackageError(f"source file is missing: {relative}") from exc
        if stat.S_ISLNK(mode):
            raise PackageError(f"links are not permitted: {relative}")
    if not stat.S_ISREG(mode):
        raise PackageError(f"source must be a regular file: {relative}")
    return path


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PackageError(f"duplicate recipe JSON key: {key}")
        result[key] = value
    return result


def load_recipe(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise PackageError("recipe must be a regular JSON file")
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise PackageError("invalid recipe JSON") from exc
    if not isinstance(recipe, dict) or type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1:
        raise PackageError("recipe schema_version must be 1")
    if type(recipe.get("frozen")) is not bool:
        raise PackageError("recipe must explicitly declare frozen=true or false")
    if not isinstance(recipe.get("python_version"), str) or not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+", recipe["python_version"]
    ):
        raise PackageError("python_version must pin a full release version")
    dependencies = recipe.get("dependencies")
    if not isinstance(dependencies, dict):
        raise PackageError("dependencies must map distributions to exact versions")
    for name, version in dependencies.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name):
            raise PackageError("invalid dependency distribution name")
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,99}", version):
            raise PackageError("dependencies must use exact versions, without operators")
    archive = recipe.get("archive")
    if not isinstance(archive, dict):
        raise PackageError("archive must be an object")
    archive_path = safe_relative(archive.get("path"))
    if not archive_path.name.endswith(".tar.gz"):
        raise PackageError("archive path must end in .tar.gz")
    entries = recipe.get("source_files")
    if not isinstance(entries, list) or not 0 < len(entries) <= MAX_FILES:
        raise PackageError("recipe needs between 1 and 20000 source files")
    destinations = set()
    sources = set()
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"source", "path", "sha256", "bytes"}:
            raise PackageError("source entries need exactly source/path/sha256/bytes")
        safe_relative(entry["source"])
        safe_relative(entry["path"])
        if entry["path"] in destinations or entry["source"] in sources:
            raise PackageError("duplicate source or archive entry")
        destinations.add(entry["path"])
        sources.add(entry["source"])
        if not valid_sha(entry["sha256"]):
            raise PackageError("source entry needs an exact SHA256")
        total += byte_count(entry["bytes"], "entry bytes", limit=MAX_FILE_BYTES)
        if total > MAX_UNPACKED_BYTES:
            raise PackageError("unpacked source bytes exceed 1 GiB")
        # Reject paths USTAR cannot encode instead of silently emitting PAX.
        info = canonical_tar_info(entry)
        try:
            info.tobuf(format=tarfile.USTAR_FORMAT, encoding="utf-8", errors="strict")
        except (ValueError, UnicodeError) as exc:
            raise PackageError(f"path cannot be encoded as USTAR: {entry['path']}") from exc
    for destination in destinations:
        if any(
            parent.as_posix() in destinations
            for parent in PurePosixPath(destination).parents
            if parent != PurePosixPath(".")
        ):
            raise PackageError("an archive file cannot also be another entry's parent")
    producers = recipe.get("producers")
    if not isinstance(producers, list):
        raise PackageError("producers must be a list")
    seen_producers = set()
    for producer in producers:
        if not isinstance(producer, dict) or set(producer) != {"path", "sha256", "args"}:
            raise PackageError("producer entries need exactly path/sha256/args")
        if producer["path"] not in PRODUCERS or producer["path"] in seen_producers:
            raise PackageError("producer must be a unique project data generator")
        seen_producers.add(producer["path"])
        if not valid_sha(producer["sha256"]):
            raise PackageError("producer needs an exact SHA256")
        arguments = producer["args"]
        if not isinstance(arguments, list) or len(arguments) > 100:
            raise PackageError("producer args must be a bounded string list")
        for argument in arguments:
            if not isinstance(argument, str) or len(argument) > 4096 or "\0" in argument:
                raise PackageError("invalid producer argument")
            if argument.split("=", 1)[0] in BLOCKED_PRODUCER_ARGS:
                raise PackageError("producers must use default directories without research or verify-only flags")
            if argument.startswith("-") and argument.split("=", 1)[0] not in ALLOWED_PRODUCER_FLAGS[producer["path"]]:
                raise PackageError("producer flags must use explicit allowed names, without argparse abbreviations")
    return recipe


def verify_producers(recipe: dict, root: Path, *, require_committed: bool = False) -> None:
    # Every generator is checked before any producer is invoked.
    for producer in recipe["producers"]:
        path = regular_source(root, producer["path"])
        if path.stat().st_size > MAX_FILE_BYTES or sha256_file(path) != producer["sha256"]:
            raise PackageError(f"producer SHA256 mismatch: {producer['path']}")
        if require_committed:
            try:
                blob = subprocess.check_output(
                    ["git", "show", f"HEAD:{producer['path']}"], cwd=root, stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as exc:
                raise PackageError(f"producer is not committed at HEAD: {producer['path']}") from exc
            if hashlib.sha256(blob).hexdigest() != producer["sha256"]:
                raise PackageError(f"committed producer SHA256 mismatch: {producer['path']}")


def verify_sources(recipe: dict, root: Path = ROOT) -> int:
    total = 0
    for entry in recipe["source_files"]:
        path = regular_source(root, entry["source"])
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise PackageError(f"source SHA256/bytes mismatch: {entry['source']}")
        total += entry["bytes"]
    return total


def verify_versions(recipe: dict) -> dict:
    if platform.python_version() != recipe["python_version"]:
        raise PackageError("runtime Python version differs from recipe pin")
    installed = {}
    for distribution, expected in recipe["dependencies"].items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise PackageError(f"required distribution is missing: {distribution}") from exc
        if actual != expected:
            raise PackageError(f"distribution version mismatch: {distribution}")
        installed[distribution] = actual
    return installed


def canonical_tar_info(entry: dict) -> tarfile.TarInfo:
    info = tarfile.TarInfo(entry["path"])
    info.type = tarfile.REGTYPE
    info.size = entry["bytes"]
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    info.mtime = 0
    return info


class DigestReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        return data


def write_archive(recipe: dict, root: Path, destination: Path) -> None:
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed,
    ):
        with tarfile.open(
            fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT, encoding="utf-8", errors="strict"
        ) as archive:
            for entry in sorted(recipe["source_files"], key=lambda item: item["path"]):
                source = regular_source(root, entry["source"])
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                with os.fdopen(os.open(source, flags), "rb") as stream:
                    current = os.fstat(stream.fileno())
                    if not stat.S_ISREG(current.st_mode) or current.st_size != entry["bytes"]:
                        raise PackageError(f"source changed before packing: {entry['source']}")
                    reader = DigestReader(stream)
                    archive.addfile(canonical_tar_info(entry), reader)
                    if (
                        reader.bytes_read != entry["bytes"]
                        or reader.digest.hexdigest() != entry["sha256"]
                        or stream.read(1)
                    ):
                        raise PackageError(f"source changed while packing: {entry['source']}")


def write_json_atomic(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".recipe-", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_git_sha(root: Path) -> str | None:
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return result if re.fullmatch(r"[0-9a-f]{40}", result) else None


def package_recipe(
    recipe_path: Path,
    output_dir: Path,
    *,
    root: Path = ROOT,
    reconstruct: bool = False,
    freeze_recipe: bool = False,
) -> dict:
    if reconstruct and freeze_recipe:
        raise PackageError("first local freeze cannot reconstruct")
    recipe_path = recipe_path.resolve()
    root = root.resolve()
    recipe = load_recipe(recipe_path)
    expected = recipe["archive"]
    if freeze_recipe:
        if recipe["frozen"] or expected.get("sha256"):
            raise PackageError("refusing to overwrite an existing recipe freeze")
    else:
        if not recipe["frozen"] or not valid_sha(expected.get("sha256")):
            raise PackageError("normal packaging requires a frozen recipe with archive SHA256")
        byte_count(expected.get("bytes"), "expected archive bytes")
        byte_count(expected.get("unpacked_bytes"), "expected unpacked bytes", limit=MAX_UNPACKED_BYTES)
    verify_producers(recipe, root, require_committed=reconstruct)
    installed = {}
    executed = []
    if reconstruct:
        if [producer["path"] for producer in recipe["producers"]] != list(PRODUCERS):
            raise PackageError("reconstruct requires committed vision/OCR, voice, then text/tools producers")
        installed = verify_versions(recipe)
        for producer in recipe["producers"]:
            subprocess.run([sys.executable, str(root / producer["path"]), *producer["args"]], cwd=root, check=True)
            executed.append(producer["path"])
    unpacked_bytes = verify_sources(recipe, root)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / PurePosixPath(expected["path"]).name
    receipt_path = final.with_name(final.name + ".receipt.json")
    if final.is_symlink() or receipt_path.is_symlink():
        raise PackageError("archive and receipt destinations must not be links")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".archive-", suffix=".tmp", dir=output_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_archive(recipe, root, temporary)
        actual = {
            "path": expected["path"],
            "sha256": sha256_file(temporary),
            "bytes": temporary.stat().st_size,
            "unpacked_bytes": unpacked_bytes,
        }
        if not freeze_recipe and any(actual[key] != expected[key] for key in ("sha256", "bytes", "unpacked_bytes")):
            raise PackageError("resulting archive SHA256/bytes/unpacked_bytes differ from frozen recipe")
        if freeze_recipe:
            recipe["archive"] = actual
            recipe["frozen"] = True
            write_json_atomic(recipe_path, recipe)
        temporary.chmod(0o644)
        temporary.replace(final)
        receipt = {
            "schema_version": 1,
            "verified": True,
            "recipe_sha256": sha256_file(recipe_path),
            "source_git_sha": source_git_sha(root),
            "archive": actual,
            "output_path": str(final),
            "source_files_verified": len(recipe["source_files"]),
            "producer_sha256_verified": len(recipe["producers"]),
            "producers_executed": executed,
            "reconstructed": reconstruct,
            "first_local_freeze": freeze_recipe,
            "runtime_python_version": platform.python_version(),
            "dependency_pins_verified": reconstruct,
            "installed_dependencies": installed,
            "zlib_version": zlib.ZLIB_VERSION,
            "tar_contract": "regular files only; sorted paths; USTAR; uid/gid=0; uname/gname=''; mode=0644; mtime=0",
            "gzip_contract": "filename=''; mtime=0; compresslevel=9",
        }
        write_json_atomic(receipt_path, receipt)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reconstruct", action="store_true")
    mode.add_argument("--freeze-recipe", action="store_true", help="first local freeze of existing sources only")
    args = parser.parse_args()
    try:
        receipt = package_recipe(
            args.recipe, args.output_dir, reconstruct=args.reconstruct, freeze_recipe=args.freeze_recipe
        )
    except (PackageError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"package verification failed: {exc}\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
