"""核對固定操作頁的版本、紀錄、官方配套與本機程式契約。"""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from scripts import fetch_natural_release as release
from tiny_perceptron import natural_ui as ui


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_bytes())


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    return {"command": args, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def original(spec):
    name, url, needles = spec
    args = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode:
        return {"id": name, "url": url, "command": args, "exit_code": result.returncode,
                "error": result.stderr.decode(errors="replace")}
    raw = result.stdout
    text = raw.decode()
    if "<html" in text.lower():
        parser = Text()
        parser.feed(text)
        text = "\n".join(parser.parts)
    lines = text.splitlines()
    excerpts = []
    for needle in needles:
        positions = [i for i, line in enumerate(lines) if needle in line]
        excerpts.append({"needle": needle, "matches": len(positions),
                         "text": "\n\n".join("\n".join(lines[max(0, i-1):i+4]) for i in positions[:2])})
    return {"id": name, "url": url, "command": args, "exit_code": 0, "accessed_on": "2026-10-20",
            "bytes": len(raw), "sha256": digest(raw), "excerpts": excerpts}


manifest_path = "docs/natural-assistant/v4/public-release.json"
manifest = release.validate_manifest(read(manifest_path))
pin = "ee17b8871cfe2ba9807b796dd11f339b5dae3b5c"
paths = ["docs/natural-assistant/v4/STUDENT.md", "docs/course-review-20261019/freeze-01/sources/natural-v4-student.md",
         "docs/course-review-20261019/freeze-04.json", manifest_path, "requirements-natural.txt",
         "scripts/fetch_natural_release.py", "tiny_perceptron/natural_assistant.py", "tiny_perceptron/natural_ui.py",
         "tests/test_natural_release.py", "tests/test_natural_base_release.py", "tests/test_natural_ui.py",
         "tiny_perceptron/selftrained/model.py", "scripts/selftrained/train.py"]
inventory = []
for path in paths:
    raw = (ROOT / path).read_bytes()
    item = {"path": path, "sha256": digest(raw), "bytes": len(raw)}
    if path not in paths[1:3]:
        old = subprocess.run(["git", "show", pin + ":" + path], cwd=ROOT, capture_output=True, check=False)
        item.update(pinned_commit=pin, pinned_exit_code=old.returncode,
                    pinned_sha256=digest(old.stdout), same_as_pinned=old.stdout == raw)
    inventory.append(item)

specs = [
    ("pytorch-install", "https://pytorch.org/get-started/previous-versions/", ["v2.8.0", "torch==2.8.0 torchvision==0.23.0", "cu128"]),
    ("venv", "https://raw.githubusercontent.com/python/cpython/v3.12.7/Doc/library/venv.rst", ["isolated", "bin", "pip"]),
    ("lfs", "https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.0/docs/man/git-lfs-config.adoc", ["GIT_LFS_SKIP_SMUDGE"]),
    ("torch-types", "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/tensors.rst", ["32-bit floating point", "16-bit floating point"]),
    ("transformers-cache", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/utils/hub.py", ["If local_files_only, don't try the HEAD call", "local_files_only (`bool`"]),
    ("hub-cache", "https://raw.githubusercontent.com/huggingface/huggingface_hub/v0.36.2/src/huggingface_hub/file_download.py", ["if local_files_only:", "Cannot find the requested files in the disk cache"]),
    ("loopback", "https://www.rfc-editor.org/rfc/rfc1122.txt", ["Internal host loopback address"]),
]
with ThreadPoolExecutor(max_workers=4) as pool:
    official = list(pool.map(original, specs))

history_root = "docs/natural-assistant/evidence/v4-research/student-selected-public-ui/"
history_paths = [history_root + name for name in ["actual-ui/report.json", "actual-ui/browser/report.json",
                 "actual-ui/disk-observation.json", "actual-ui/observer.jsonl", "anonymous-fetch/receipt.json",
                 "executed-harness/ui_observer.py.source.txt", "executed-harness/run_ui_smoke.py.source.txt"]]
cache_path = "docs/natural-assistant/evidence/v4-research/student-local-cache-prepare/actual-prepare/result.json"
history_paths.append(cache_path)
history_inventory = [{"path": p, "sha256": digest((ROOT / p).read_bytes())} for p in history_paths]
run = read(history_paths[0])
browser = read(history_paths[1])
disk = read(history_paths[2])
events = [json.loads(line) for line in (ROOT / history_paths[3]).read_text().splitlines()]
cache = read(cache_path)
snapshot_files = [dict(model=s["model"], **f) for s in cache["snapshots"].values() for f in s["files"]]
metrics = {
    "source_records": history_inventory,
    "snapshot_files": snapshot_files,
    "snapshot_count": len(snapshot_files),
    "snapshot_bytes": sum(f["bytes"] for f in snapshot_files),
    "snapshot_gib": sum(f["bytes"] for f in snapshot_files) / 2**30,
    "public_files_count": len(manifest["files"]),
    "public_files_bytes": sum(f["bytes"] for f in manifest["files"]),
    "server_resources": run["server_cpu_resources"],
    "rss_gib": run["server_cpu_resources"]["maximum_resident_set_kib"] / 2**20,
    "ready_seconds": run["ready_seconds"], "wall_seconds": run["wall_seconds"],
    "chat_seconds": browser["chat_seconds"], "asr_including_load": browser["asr_seconds_including_load"],
    "recorded_environment": {k: run[k] for k in ["device", "dtype", "torch_threads", "torch_interop_threads", "versions"] if k in run},
    "observer_ready": events[0],
    "load_options": [e for e in events if e["kind"] in ["load_core_begin", "load_asr_begin"]],
    "disk": disk,
    "history_lengths": [len(e["row"]["history"]) for e in events if e["kind"] == "generate_begin"],
    "generation_scores": [e["result"]["score"] for e in events if e["kind"] == "generate_end"],
    "reset": [e for e in events if e["kind"] == "operation_end" and e["route"] == "/api/reset"],
    "shutdown_removed_uploads": run["shutdown_removed_new_upload_directories"],
    "historical_anonymous_fetch": read(history_paths[4]),
    "scope": "重算已保存的單次 Linux CPU 瀏覽器操作紀錄，未重跑模型推論或效能測量。",
}

with tempfile.TemporaryDirectory(prefix="audit-work-", dir=HERE) as temporary:
    work = Path(temporary)
    downloaded = {}
    live_files = []
    for item in manifest["files"]:
        url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{item['path']}"
        args = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
        result = subprocess.run(args, capture_output=True, check=False)
        observation = {"url": url, "command": args, "exit_code": result.returncode,
                       "bytes": len(result.stdout), "sha256": digest(result.stdout)}
        observation["matches_manifest"] = result.returncode == 0 and len(result.stdout) == item["bytes"] and digest(result.stdout) == item["sha256"]
        live_files.append(observation)
        if not observation["matches_manifest"]:
            raise RuntimeError("匿名取得的公開文件與清單不同。")
        target = work / item["output"]
        target.write_bytes(result.stdout)
        downloaded[item["output"]] = target
    calls = []

    def downloader(repo, path, **kwargs):
        calls.append({"repo": repo, "path": path, **kwargs})
        return downloaded[Path(path).name]

    folder = release.fetch_release(manifest, work / "release", downloader=downloader)
    verified = str(release.verify_release(manifest, folder).relative_to(work))
    (folder / "extra.txt").write_text("額外檔案")
    try:
        release.verify_release(manifest, folder)
        extra_rejected = False
    except ValueError:
        extra_rejected = True
    (folder / "extra.txt").unlink()
    protected_before = {p.name: digest(p.read_bytes()) for p in folder.iterdir()}
    try:
        release.fetch_release(manifest, folder, downloader=downloader)
        existing_rejected = False
    except ValueError:
        existing_rejected = True
    protected_after = {p.name: digest(p.read_bytes()) for p in folder.iterdir()}
    runtime_guard = None
    try:
        release.check_runtime(manifest)
    except ValueError as error:
        runtime_guard = str(error)
    modes = {}
    for device, dtype in [("cpu", None), ("cuda", None), ("cuda", "float16")]:
        options = release.student_options(manifest, folder, device=device, dtype=dtype, local_files_only=True)
        modes[device + ":" + str(dtype)] = {k: str(getattr(options, k)) for k in
                                          ["model", "model_revision", "asr_model", "asr_revision", "adapter", "device", "dtype", "local_files_only"]}
    image_formats = {}
    for fmt, suffix in [("PNG", "png"), ("JPEG", "jpg"), ("WEBP", "webp")]:
        stream = io.BytesIO()
        Image.new("RGB", (12, 16)).save(stream, format=fmt)
        raw = stream.getvalue()
        accepted = ui.upload_contents({"kind": "image", "filename": "photo."+suffix, "base64": base64.b64encode(raw).decode()})
        image_formats[fmt] = {"accepted": accepted[0] == raw, "media_type": accepted[2]}
    audio_formats = {}
    for fmt, suffix in [("WAV", "wav"), ("FLAC", "flac"), ("MP3", "mp3"), ("OGG", "ogg")]:
        stream = io.BytesIO()
        sf.write(stream, np.zeros(1600), 16000, format=fmt)
        raw = stream.getvalue()
        accepted = ui.upload_contents({"kind": "audio", "filename": "voice."+suffix, "base64": base64.b64encode(raw).decode()})
        audio_formats[fmt] = {"accepted": accepted[0] == raw, "media_type": accepted[2]}
    tests = command([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(work / "pytest"),
                     "tests/test_natural_release.py", "tests/test_natural_base_release.py", "tests/test_natural_ui.py",
                     "-k", "not publisher and not approval"])
    local = {"live_public_files": live_files, "fetch_calls": calls, "native_verify": verified,
             "extra_file_rejected": extra_rejected, "existing_folder_rejected": existing_rejected,
             "existing_folder_preserved": protected_before == protected_after, "student_options": modes,
             "runtime_guard_on_this_host": runtime_guard, "image_formats": image_formats, "audio_formats": audio_formats,
             "limits": {"upload_bytes": ui.MAX_UPLOAD_BYTES, "upload_decimal_mb": ui.MAX_UPLOAD_BYTES / 1_000_000,
                        "image_pixels": ui.MAX_IMAGE_PIXELS, "float32_bits": torch.zeros(1, dtype=torch.float32).element_size()*8},
             "pytest": tests}

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
    "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(),
                    "device": "cpu", "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__},
    "head": command(["git", "rev-parse", "HEAD"]), "inventory": inventory, "official_originals": official,
    "recorded_trial": metrics, "local_execution": local,
    "cli_list": command([sys.executable, "scripts/fetch_natural_release.py", "--manifest", manifest_path, "--list"]),
    "limitations": ["未在本機安裝 Python 3.12 的正式配套；本機 Python 3.13 的拒絕行為另有記錄。",
                    "未重跑模型權重下載、真模型生成、Linux CPU 計時或 CUDA 路線；歷史實測以原始紀錄核對。",
                    "HTTP 測試注入模型執行器，只支持路由、狀態與清除行為。"],
}
print(json.dumps(output, ensure_ascii=False, indent=2))
