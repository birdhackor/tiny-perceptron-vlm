"""Reuse the known immutable local model cache and run the actual offline prepare CLI."""

import hashlib
import json
import os
import resource
import shutil
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/natural-extension/student-base-cache/hf/models--Qwen--Qwen3-VL-2B-Instruct"
CACHE = ROOT / "outputs/natural-v4/student-base-cache/hf"
TARGET = CACHE / SOURCE.name
CORE_PIN = "89644892e4d85e24eaac8bacfd4f463576704203"
ASR_PIN = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


def fingerprint(path):
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
    return {"bytes": path.stat().st_size, "sha256": sha256.hexdigest()}


def tree(root):
    result = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            link = os.readlink(path)
            if Path(link).is_absolute() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("Known cache must have intact relative snapshot links inside its own model directory")
            result[name] = {"kind": "symlink", "target": link}
        elif path.is_dir():
            result[name] = {"kind": "directory"}
        elif stat.S_ISREG(path.stat().st_mode):
            result[name] = {"kind": "regular", **fingerprint(path)}
        else:
            raise ValueError("Unexpected special file in the known cache")
    return result


def main():
    if TARGET.exists() or TARGET.is_symlink():
        raise FileExistsError("The Qwen target must be new; this operation never overwrites model cache content")
    started = datetime.now(UTC).isoformat()
    code_paths = (ROOT / "scripts/natural_assistant.py", ROOT / "tiny_perceptron/natural_assistant.py")
    code_before = {path.relative_to(ROOT).as_posix(): fingerprint(path) for path in code_paths}
    initial_source = tree(SOURCE)
    if not (SOURCE / "snapshots" / CORE_PIN / "model.safetensors").is_file():
        raise ValueError("The previously known official core pin is absent")
    if not (CACHE / "models--openai--whisper-large-v3-turbo/snapshots" / ASR_PIN / "model.safetensors").is_file():
        raise ValueError("The previously known official turbo pin is absent")
    resources_before = {
        "disk_usage": shutil.disk_usage(ROOT)._asdict(),
        "meminfo": Path("/proc/meminfo").read_text(),
    }
    copied_at = time.monotonic()
    shutil.copytree(SOURCE, TARGET, symlinks=True, copy_function=os.link)
    copy_seconds = time.monotonic() - copied_at
    links = []
    for name, item in initial_source.items():
        source, target = SOURCE / name, TARGET / name
        if item["kind"] == "regular":
            first, second = source.stat(), target.stat()
            if (first.st_dev, first.st_ino) != (second.st_dev, second.st_ino):
                raise ValueError("Copied immutable blob is not a real hardlink")
            links.append({"path": name, "bytes": item["bytes"], "same_device_and_inode": True})
        elif item["kind"] == "symlink" and os.readlink(source) != os.readlink(target):
            raise ValueError("Snapshot symlink text changed")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    command = [
        str(ROOT / ".venv-natural/bin/python"), str(ROOT / "scripts/natural_assistant.py"), "prepare",
        "--asr-variant", "turbo", "--cache-dir", str(CACHE), "--local-files-only",
        "--device", "cpu", "--dtype", "float32", "--output", str(EVIDENCE / "actual-prepare"),
    ]
    environment = dict(os.environ)
    environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        environment.pop(key, None)
    wall = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
    wall_seconds = time.monotonic() - wall
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    (EVIDENCE / "prepare.stdout.json").write_text(result.stdout)
    (EVIDENCE / "prepare.stderr.txt").write_text(result.stderr)
    source_after = tree(SOURCE)
    target_after = tree(TARGET)
    code_after = {path.relative_to(ROOT).as_posix(): fingerprint(path) for path in code_paths}
    report = {
        "scope": "Actual offline known-cache integrity preparation; not a fresh anonymous download, model load, generation or scoring run",
        "started_at_UTC": started,
        "finished_at_UTC": datetime.now(UTC).isoformat(),
        "command": command,
        "exit_code": result.returncode,
        "copy_method": "shutil.copytree with os.link for immutable regular blobs and symlinks=True",
        "core_source": str(SOURCE.relative_to(ROOT)),
        "combined_cache": str(CACHE.relative_to(ROOT)),
        "core_pin": CORE_PIN,
        "asr_pin": ASR_PIN,
        "copy_seconds": copy_seconds,
        "hardlinks": links,
        "source_tree_and_bytes_unchanged": initial_source == source_after,
        "target_tree_and_bytes_identical_to_known_source": initial_source == target_after,
        "source_before": initial_source,
        "source_after": source_after,
        "source_code_before": code_before,
        "source_code_after": code_after,
        "source_code_unchanged": code_before == code_after,
        "prepare_wall_seconds": wall_seconds,
        "prepare_user_CPU_seconds": after.ru_utime - before.ru_utime,
        "prepare_system_CPU_seconds": after.ru_stime - before.ru_stime,
        "prepare_max_RSS_KiB": after.ru_maxrss,
        "resources_before": resources_before,
        "resources_after": {"disk_usage": shutil.disk_usage(ROOT)._asdict(), "meminfo": Path("/proc/meminfo").read_text()},
        "HF_HUB_OFFLINE": True,
        "HF_DATASETS_OFFLINE": True,
        "local_files_only": True,
        "no_network_download_claimed": True,
        "model_generation_runs": 0,
        "model_scoring_or_test_runs": 0,
        "GPU_started": False,
    }
    if result.returncode == 0:
        prepared = json.loads((EVIDENCE / "actual-prepare/result.json").read_bytes())
        if prepared["status"] != "completed":
            raise ValueError("Actual prepare did not complete")
        for key, pin in (("core", CORE_PIN), ("asr", ASR_PIN)):
            snapshot = prepared["snapshots"][key]
            if snapshot["revision"] != pin or Path(snapshot["snapshot"]).name != pin:
                raise ValueError("Actual prepare returned an unexpected official pin")
            for file in snapshot["files"]:
                resolved = (Path(snapshot["snapshot"]) / file["name"]).resolve()
                if len(resolved.name) == 64 and resolved.name != file["sha256"]:
                    raise ValueError("Official LFS immutable blob SHA does not match actual prepared bytes")
        report["prepared_snapshots"] = prepared["snapshots"]
        report["prepared_files"] = sum(len(s["files"]) for s in prepared["snapshots"].values())
        report["prepared_bytes"] = sum(f["bytes"] for s in prepared["snapshots"].values() for f in s["files"])
        report["actual_prepare_all_file_SHA256_recorded"] = True
        report["official_immutable_weight_blob_SHA256_matches"] = True
    (EVIDENCE / "cache-prepare-receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if result.returncode or not report["source_tree_and_bytes_unchanged"] or not report["target_tree_and_bytes_identical_to_known_source"] or not report["source_code_unchanged"]:
        raise RuntimeError("Offline prepare or immutable-cache preservation check failed; inspect saved actual receipts")
    print(json.dumps({k: report[k] for k in ("scope", "exit_code", "prepared_files", "prepared_bytes", "prepare_wall_seconds", "prepare_max_RSS_KiB", "source_tree_and_bytes_unchanged", "target_tree_and_bytes_identical_to_known_source")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
