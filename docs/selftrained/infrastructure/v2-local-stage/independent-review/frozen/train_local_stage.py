#!/usr/bin/env python3
"""Run one genuine local training stage and preserve its same-directory provenance.

Use an extracted frozen data package and trusted checkpoints created by this
wrapper. Published safetensors exports support inference, not training init.
Trainer arguments follow ``--``; each invocation requires a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now():
    return datetime.now(UTC).isoformat()


def verified_file(root, item):
    relative = Path(item["path"])
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or path.stat().st_size != item["bytes"]
        or digest(path) != item["sha256"]
    ):
        raise ValueError(f"File differs from its recorded bytes/SHA: {relative}")
    return path


def local_source(path, manifest_sha, resume):
    if path.is_symlink() or path.name not in (("best.pt", "latest.pt") if resume else ("best.pt",)):
        raise ValueError("Use an own local best.pt for fresh init or saved .pt for exact resume")
    execution = json.loads((path.parent / "execution.json").read_text())
    receipt = json.loads((path.parent / "receipt.json").read_text())
    if (
        execution.get("backend") != "local-subprocess"
        or execution.get("manifest_sha256") != manifest_sha
        or execution.get("status") not in (("completed", "failed") if resume else ("completed",))
        or (not resume and execution.get("returncode") != 0)
        or any(receipt.get(key) != value for key, value in execution.items())
    ):
        raise ValueError("Source needs its genuine matching local execution/receipt; incomplete init is forbidden")
    entries = {item["path"]: item for item in receipt["files"]}
    if (
        len(entries) != len(receipt["files"])
        or not {path.name, "execution.json", "train-receipt.json"} <= entries.keys()
    ):
        raise ValueError("Source receipt lacks its checkpoint or genuine execution/training metadata")
    for item in entries.values():
        verified_file(path.parent, item)
    training = json.loads((path.parent / "train-receipt.json").read_text())
    if not resume and (
        training.get("completed_requested_steps") is not True or training.get("interrupted") is not False
    ):
        raise ValueError("Interrupted training cannot be promoted to fresh stage init")
    return {
        "path": str(path),
        "sha256": entries[path.name]["sha256"],
        "run_id": execution["run_id"],
        "revision": execution["revision"],
        "receipt_sha256": digest(path.parent / "receipt.json"),
        "mode": "exact_resume" if resume else "fresh_stage_init",
    }


def source_identity(repo):
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True).stdout

    revision = git("rev-parse", "HEAD").decode().strip()
    files = (
        git(
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "scripts/selftrained",
            "tiny_perceptron",
            "pyproject.toml",
            "uv.lock",
        )
        .decode()
        .splitlines()
    )
    if "scripts/selftrained/train.py" not in files:
        raise ValueError("The trainer must be committed in the selected local repository")
    hashes = {}
    for relative in files:
        path = repo / relative
        if path.read_bytes() != git("show", f"{revision}:{relative}"):
            raise ValueError(f"Training source differs from recorded Git revision: {relative}")
        hashes[relative] = digest(path)
    return revision, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", help="Optional external pin of the frozen manifest bytes")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    wrapper = parser.parse_args(argv)
    forwarded = wrapper.trainer_args
    if not forwarded or forwarded.pop(0) != "--":
        parser.error("Place existing trainer arguments after --")
    managed = {"--records", "--asset-dir", "--config", "--export-inference"}
    if any(arg.split("=", 1)[0] in managed for arg in forwarded):
        parser.error("records, asset-dir, config and export-inference are provided by the frozen manifest")
    repo, data_root = wrapper.repo_root.resolve(), wrapper.data_root.resolve()
    manifest_sha = digest(wrapper.manifest)
    if wrapper.manifest_sha256 is not None and wrapper.manifest_sha256 != manifest_sha:
        raise ValueError("Manifest differs from its external SHA256 pin")
    manifest = json.loads(wrapper.manifest.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("initialization") != "random":
        raise ValueError("Require a frozen from-random training manifest")
    declared = manifest["records"] + manifest.get("assets", [])
    if len({item["path"] for item in declared}) != len(declared):
        raise ValueError("Manifest repeats a declared file")
    if len({Path(item["path"]).name for item in manifest["records"]}) != len(manifest["records"]):
        raise ValueError("Trainer record basenames must be unique")
    for item in declared:
        verified_file(data_root, item)
    revision, source_hashes = source_identity(repo)
    spec = importlib.util.spec_from_file_location("local_stage_trainer", repo / "scripts/selftrained/train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    record_args = [part for item in manifest["records"] for part in ("--records", str(data_root / item["path"]))]
    managed_args = record_args + ["--asset-dir", str(data_root), "--export-inference"]
    args = trainer.parser().parse_args(forwarded + managed_args)
    if args.context != manifest["model_config"].get("max_length", 512):
        raise ValueError("Context must retain the frozen manifest model config")
    if args.resume and args.init_checkpoint:
        raise ValueError("resume and init-checkpoint are mutually exclusive")
    output = Path(args.output_dir).resolve()
    source = None
    for name in ("resume", "init_checkpoint"):
        if getattr(args, name):
            path = Path(getattr(args, name)).absolute()
            source = local_source(path, manifest_sha, name == "resume")
            forwarded += ["--" + name.replace("_", "-"), str(path)]
    output.mkdir(parents=True, exist_ok=False)
    config = output / "input-model-config.json"
    write_json(config, manifest["model_config"])
    final_args = forwarded + managed_args + ["--config", str(config), "--output-dir", str(output)]
    job = vars(trainer.parser().parse_args(final_args))
    command = [sys.executable, "-u", str(repo / "scripts/selftrained/train.py"), *final_args]
    execution = {
        "backend": "local-subprocess",
        "run_id": "local-" + uuid.uuid4().hex,
        "revision": revision,
        "code_sha256": source_hashes,
        "wrapper_sha256": digest(__file__),
        "manifest_sha256": manifest_sha,
        "manifest_declared_files": len(declared),
        "stage": args.stage,
        "job": job,
        "command": command,
        "cwd": str(repo),
        "parent": source,
        "started_at": utc_now(),
        "status": "running",
    }
    write_json(output / "execution.json", execution)
    received_signals = []
    child = None

    def stop(signum, frame):
        received_signals.append(signum)
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
    error = None
    returncode = None
    try:
        with (output / "runner.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = child.wait()
    except OSError as exc:
        error = str(exc)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    training_path = output / "train-receipt.json"
    training = json.loads(training_path.read_text()) if training_path.is_file() else {}
    required = (
        "best.pt",
        "latest.pt",
        "inference-manifest.json",
        "model-config.json",
        "tokenizer.json",
        "model.safetensors",
    )
    complete = (
        returncode == 0
        and training.get("completed_requested_steps") is True
        and training.get("interrupted") is False
        and training.get("steps") == args.steps
        and training.get("selected_checkpoint_available") is True
        and training.get("inference_exported") is True
        and all((output / name).is_file() for name in required)
    )
    execution.update(
        status="completed" if complete else "failed",
        returncode=returncode,
        interrupted=bool(received_signals) or training.get("interrupted", False),
        received_signals=received_signals,
        finished_at=utc_now(),
        error=error,
    )
    write_json(output / "execution.json", execution)
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "receipt.json"
    ]
    write_json(output / "receipt.json", {**execution, "files": files})
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
