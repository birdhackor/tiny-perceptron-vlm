"""Bounded fresh review: native release operations, injected UI state, raw-run audit.

No model load/install/training; two small public project documents are retrieved.
Injected model/ASR functions below establish state wiring, not model quality.
"""
import base64
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from tiny_perceptron import natural_assistant as core, natural_ui as ui

spec = importlib.util.spec_from_file_location("fresh_fetch", ROOT / "scripts/fetch_natural_release.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
research = ROOT / "outputs/natural-v4/factual-research/20.2"
research.mkdir(parents=True, exist_ok=True)
manifest = fetch.read_json(ROOT / "docs/natural-assistant/v4/public-release.json")
fetch.validate_manifest(manifest)
env = {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu", "cuda_available": str(torch.cuda.is_available())}
for name in ("numpy", "pillow", "soundfile", "huggingface-hub"):
    env[name] = importlib.metadata.version(name)
out = {"scope": "Own bounded CPU probes plus independent audit of existing original trial; no own model inference or install", "environment": env}
assert torch.__version__ == "2.14.1+cpu"

# Read-only comparison of exact student commit and current program bytes.
pin = "59a1eda4ed7b6e8609892ec2b9013c821ac93e69"
out["pinned_files"] = []
for path in ["scripts/fetch_natural_release.py", "tiny_perceptron/natural_assistant.py", "tiny_perceptron/natural_ui.py", "requirements-natural.txt", "docs/natural-assistant/v4/public-release.json"]:
    result = subprocess.run(["git", "show", f"{pin}:{path}"], cwd=ROOT, capture_output=True, check=True)
    data = (ROOT / path).read_bytes()
    assert data == result.stdout
    out["pinned_files"].append({"path": path, "sha256": hashlib.sha256(data).hexdigest(), "same_pinned_bytes": True})
out["git"] = subprocess.run(["git", "--version"], capture_output=True, text=True, check=True).stdout.strip()
out["python3.12"] = subprocess.run(["python3.12", "--version"], capture_output=True, text=True, check=True).stdout.strip()
requirements = {line.split("==")[0]: line.split("==")[1] for line in (ROOT / "requirements-natural.txt").read_text().splitlines() if line and not line.startswith("#")}
assert requirements == manifest["dependency_versions"]
assert fetch.sha256(fetch.REQUIREMENTS) == manifest["requirements_sha256"]
out["requirements"] = requirements
section = (Path(__file__).parent / "reviewed-section.raw.md").read_text()
import re
blocks = re.findall(r"```bash\n(.*?)```", section, re.S)
for block in blocks:
    subprocess.run(["bash", "-n"], input=block, text=True, check=True, capture_output=True)
out["bash_blocks_syntax_checked"] = len(blocks)
notebook = json.loads((ROOT / "notebooks/20/20.2.ipynb").read_text())
assert all(cell["cell_type"] == "markdown" for cell in notebook["cells"])
out["notebook_cell_types"] = [cell["cell_type"] for cell in notebook["cells"]]
result = subprocess.run([sys.executable, "scripts/fetch_natural_release.py", "--manifest", "docs/natural-assistant/v4/public-release.json", "--list"], cwd=ROOT, capture_output=True, text=True, check=True)
out["native_list"] = {"exit_code": result.returncode, "stdout": json.loads(result.stdout), "stderr": result.stderr}

# Fetch exactly the manifest's 9,262 document bytes, without authentication.
retrievals = []
def downloader(repo, filename, *, revision, token):
    assert repo == manifest["repo"] and revision == manifest["revision"] and token is False
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(12000)
        status = response.status
    path = research / (Path(filename).name + ".anonymous.raw")
    path.write_bytes(data)
    retrievals.append({"url": url, "status": status, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "authentication_headers": []})
    return path
with tempfile.TemporaryDirectory(dir=research) as temp:
    target = Path(temp) / "release"
    fetch.fetch_release(manifest, target, downloader=downloader)
    fetch.verify_release(manifest, target)
    cli = subprocess.run([sys.executable, "scripts/fetch_natural_release.py", "--manifest", "docs/natural-assistant/v4/public-release.json", "--output", str(target), "--verify"], cwd=ROOT, capture_output=True, text=True, check=True)
    opt = fetch.student_options(manifest, target, device="cpu", dtype="float32")
    assert opt.adapter is None and opt.model_revision == manifest["base_model"]["revision"] and opt.asr_revision == manifest["asr_model"]["revision"]
    out["student_options"] = {key: str(getattr(opt, key)) for key in ["model", "model_revision", "asr_model", "asr_revision", "adapter", "device", "dtype", "local_files_only"]}
    (target / "README.md").write_bytes(b"changed")
    try:
        fetch.verify_release(manifest, target)
    except ValueError as error:
        out["tamper_rejected"] = str(error)
    else:
        raise AssertionError("Modified document accepted")
out["anonymous_document_retrievals"] = retrievals
out["native_verify"] = {"exit_code": cli.returncode, "stderr": cli.stderr, "stdout_scope": "Temporary verified release path"}
try:
    fetch.check_runtime(manifest)
except ValueError as error:
    out["current_environment_gate"] = str(error)
else:
    raise AssertionError("Current teaching venv must not be confused with student Python 3.12 / Torch 2.8")

# Actual UI operations with explicitly injected deterministic model/ASR runners.
calls, loads = [], []
def generated(model, processor, row, data_root, options):
    calls.append(copy.deepcopy(row))
    core.messages_for(row, data_root)
    return {"prediction": "synthetic reply"}
def loaded(options):
    loads.append("asr")
    return object(), object()
def transcribed(model, processor, path):
    assert path.is_file()
    return {"transcript": "我吃辣", "audio_sha256": core.sha256(path)}
def upload(server, session, name, kind, data):
    return server.operation("/api/upload", {"session": session, "filename": name, "kind": kind, "base64": base64.b64encode(data).decode()})["asset"]
with tempfile.TemporaryDirectory(dir=research) as temp, patch.object(core, "generate", generated), patch.object(core, "load_asr", loaded), patch.object(core, "transcribe", transcribed):
    server = ui.create_server(None, None, opt, Path(temp), port=0)
    upload_dir = Path(server.upload_directory.name)
    session = server.operation("/api/session", {})["session"]
    server.operation("/api/chat", {"session": session, "prompt": "你好"})
    image = io.BytesIO(); Image.new("RGB", (2, 2), "red").save(image, format="PNG")
    photo = upload(server, session, "photo.png", "image", image.getvalue())
    server.operation("/api/chat", {"session": session, "prompt": "照片裡有哪些東西？", "image": photo})
    audio = io.BytesIO(); sf.write(audio, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    speech = upload(server, session, "short.wav", "audio", audio.getvalue())
    before = len(calls)
    transcript = server.operation("/api/transcribe", {"session": session, "audio": speech})
    assert len(calls) == before and len(server.sessions[session]["history"]) == 4 and loads == ["asr"]
    correction = server.operation("/api/chat", {"session": session, "prompt": "我不吃辣", "speech": speech})
    server.operation("/api/chat", {"session": session, "prompt": "追問"})
    assert calls[2]["user"] == "我不吃辣" and correction["asr"]["transcript"] == "我吃辣" and correction["asr"]["corrected"] is True
    assert core.row_assets(calls[2]) == core.row_assets(calls[3]) and len(core.row_assets(calls[3])) == 1
    before_reset = len(server.sessions[session]["history"])
    server.operation("/api/reset", {"session": session})
    assert server.sessions[session] == {"history": [], "assets": {}, "transcriptions": {}}
    assert not list(upload_dir.iterdir())
    upload(server, session, "photo.png", "image", image.getvalue())
    server.server_close()
    assert not upload_dir.exists()
out["injected_ui_probe"] = {"chat_calls": len(calls), "lazy_asr_loads": len(loads), "history_before_calls": [len(c["history"]) for c in calls], "transcribe_did_not_chat": True, "submitted_text": correction["user"], "original_transcript": correction["asr"]["transcript"], "history_image_persisted": True, "history_messages_before_reset": before_reset, "reset_and_server_close_removed_uploads": True, "loopback": server.server_address[0], "default_port_source": 8766}

# Exercise native serve's KeyboardInterrupt/finally path without model loading.
with tempfile.TemporaryDirectory(dir=research) as temp:
    opt.data_root = Path(temp) / "ui"
    cached = Path(temp) / "cached-weight-sentinel"
    cached.write_bytes(b"cache retained")
    lifecycle = {}
    original_create = ui.create_server
    def interrupted_server(*args, **kwargs):
        instance = original_create(*args, **kwargs)
        lifecycle["upload_dir"] = Path(instance.upload_directory.name)
        def interrupt():
            session = instance.operation("/api/session", {})["session"]
            upload(instance, session, "photo.png", "image", image.getvalue())
            raise KeyboardInterrupt
        instance.serve_forever = interrupt
        return instance
    with patch.object(core, "load_core", lambda *args, **kwargs: (None, None)), patch.object(ui, "create_server", interrupted_server), patch("builtins.print"):
        ui.serve(opt, port=0)
    assert not lifecycle["upload_dir"].exists() and cached.read_bytes() == b"cache retained"
out["native_serve_interrupt_probe"] = "Native serve returned after injected KeyboardInterrupt and deleted uploads; separate cache sentinel retained. No model inference."

# Verify the linked guide's exact upload limits with tiny synthetic payloads.
audio30 = io.BytesIO(); sf.write(audio30, np.zeros(480000, dtype=np.float32), 16000, format="WAV")
ui.upload_contents({"filename": "30.wav", "kind": "audio", "base64": base64.b64encode(audio30.getvalue()).decode()})
audio_long = io.BytesIO(); sf.write(audio_long, np.zeros(480001, dtype=np.float32), 16000, format="WAV")
try:
    ui.upload_contents({"filename": "long.wav", "kind": "audio", "base64": base64.b64encode(audio_long.getvalue()).decode()})
except ValueError as error:
    out["over30_seconds_rejected"] = str(error)
else:
    raise AssertionError("Overlong audio accepted")
out["upload_limits"] = {"bytes": ui.MAX_UPLOAD_BYTES, "megabytes_decimal": ui.MAX_UPLOAD_BYTES / 10**6, "image_pixels": ui.MAX_IMAGE_PIXELS, "audio_seconds": 30, "30_seconds_accepted": True, "image_formats": sorted(ui.IMAGE_TYPES), "audio_suffixes": sorted(ui.AUDIO_TYPES)}

# Independent audit, recomputed from the original raw observer and browser records.
base = ROOT / "docs/natural-assistant/evidence/v4-research/student-selected-public-ui"
report = json.loads((base / "actual-ui/report.json").read_text())
browser = json.loads((base / "actual-ui/browser/report.json").read_text())
obs = [json.loads(line) for line in (base / "actual-ui/observer.jsonl").read_text().splitlines()]
begins = [o for o in obs if o["kind"] == "generate_begin"]
ends = [o for o in obs if o["kind"] == "generate_end"]
asr = [o for o in obs if o["kind"] == "transcribe_end"]
chat = [r for r in browser["requests"] if r["route"] == "/api/chat"]
assert len(begins) == len(ends) == len(chat) == 4 and len(asr) == 1
assert [len(o["row"]["history"]) for o in begins] == [0, 2, 4, 6]
assert all(r["response"]["prediction"] == o["result"]["prediction"] for r, o in zip(chat, ends))
assert all(r["status"] == 200 for r in browser["requests"])
assert all(o["result"]["score"] is None for o in ends)
assert chat[2]["response"]["asr"]["submitted_text"] == begins[2]["row"]["user"]
assert chat[2]["response"]["asr"]["transcript"] == asr[0]["result"]["transcript"]
image_sha = report["inputs"]["image"]["sha256"]
assert all(any(a["sha256"] == image_sha for a in o["assets"]) for o in begins[1:])
reset = next(o for o in obs if o["kind"] == "operation_end" and o["route"] == "/api/reset")
assert reset["state"] == {"history_messages": 0, "assets": 0, "transcriptions": 0, "asset_files": []}
native_options = next(o["options"] for o in obs if o["kind"] == "load_core_begin")
assert native_options["adapter"] == "None" and native_options["dtype"] == "float32" and native_options["device"] == "cpu"
for source in report["sources_before"]:
    if source["path"] in {p["path"] for p in out["pinned_files"]}:
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
traffic = json.loads((base / "anonymous-fetch/fetch.http.json").read_text())["requests"]
assert sum(r["method"] == "GET" and r["status"] == 200 for r in traffic) == 2
assert all(r["authentication_headers"] == [] for r in traffic)
prepared = json.loads((ROOT / "docs/natural-assistant/evidence/v4-research/student-local-cache-prepare/actual-prepare/result.json").read_text())
snapshots = prepared["snapshots"]
files = [f for s in snapshots.values() for f in s["files"]]
assert len(files) == 23 and sum(f["bytes"] for f in files) == 5889111977
resource = next(o for o in obs if o["kind"] == "process_resources")
out["original_trial_audit"] = {"native_model_inference_replicated_by_reviewer": False, "raw_inputs": report["inputs"], "original_dependencies": report["dependencies"], "original_source_commit": report["git_revision"], "original_cpu_threads": [report["torch_threads"], report["torch_interop_threads"]], "chat_calls": len(chat), "asr_calls": len(asr), "history_lengths_before_chat": [len(o["row"]["history"]) for o in begins], "chat_input_tokens": [o["result"]["input_tokens"] for o in ends], "chat_generated_tokens_including_eos": [len(o["result"]["generated_token_ids"]) for o in ends], "asr_tokens_including_eos": len(asr[0]["result"]["generated_token_ids"]), "original_transcript": asr[0]["result"]["transcript"], "submitted_correction": begins[2]["row"]["user"], "photo_persisted_by_input_sha": image_sha, "reset_state": reset["state"], "browser_version": browser["browser_version"], "chat_seconds_rounded": [round(x, 2) for x in browser["chat_seconds"]], "asr_seconds_rounded": round(browser["asr_seconds_including_load"], 2), "ready_seconds_rounded": round(report["ready_seconds"], 2), "whole_harness_seconds_rounded": round(report["wall_seconds"], 2), "model_snapshot_file_count": len(files), "model_snapshot_bytes": sum(f["bytes"] for f in files), "model_snapshot_gib": sum(f["bytes"] for f in files) / 2**30, "public_file_bytes": sum(f["bytes"] for f in manifest["files"]), "server_rss_kib": resource["maximum_resident_set_kib"], "server_rss_gib": resource["maximum_resident_set_kib"] / 2**20, "scope": "Recomputed original record counts/conversions; original existing official model cache, not fresh weight download, benchmark, quality scoring, GPU or own model replication"}
out["result"] = "All bounded assertions passed"
print(json.dumps(out, ensure_ascii=False, indent=2))
