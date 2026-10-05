"""Bounded CPU contract checks and recomputation of existing raw measurements.

No model weights, model inference, training, or new quality evaluation.
Stand-ins exercise only loader argument forwarding and UI state transitions.
"""
import ast
import base64
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import types
from unittest.mock import patch

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sys.path.insert(0, str(ROOT))
from scripts import fetch_natural_release as release
from tiny_perceptron import natural_assistant as core
from tiny_perceptron import natural_ui as ui
from PIL import Image
import torch

torch.set_num_threads(1)
assert not torch.cuda.is_available() and torch.version.cuda is None
manifest = release.read_json(ROOT / "docs/natural-assistant/v4/public-release.json")
release.validate_manifest(manifest)
environment = {
    "python": sys.version, "executable": sys.executable, "device": "cpu",
    "torch": str(torch.__version__), "cuda_build": str(torch.version.cuda),
    "installed_dependencies": {}, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
    "scope": "Current .venv is Python 3.13 / torch 2.14 CPU, not the prescribed full-model environment",
}
for name in manifest["dependency_versions"]:
    try:
        environment["installed_dependencies"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        environment["installed_dependencies"][name] = "not installed"
(ART / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

commands = []
def run(argv, name, **kwargs):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=30, **kwargs)
    (ART / (name + ".stdout.txt")).write_bytes(result.stdout)
    (ART / (name + ".stderr.txt")).write_bytes(result.stderr)
    commands.append({"argv": list(map(str, argv)), "command": shlex.join(list(map(str, argv))),
                     "exit_code": result.returncode, "stdout": name + ".stdout.txt",
                     "stderr": name + ".stderr.txt"})
    assert result.returncode == 0, (name, result.returncode, result.stderr)
    return result

# Preserve and syntax-check every original Bash fence; do not clone/install/serve models.
section = (ART / "section.md").read_bytes()
fences = []
opened = False
for line in section.splitlines(keepends=True):
    if line.startswith(b"```bash"):
        opened = True
        fence = []
    elif opened and line.startswith(b"```"):
        opened = False
        raw = b"".join(fence)
        p = ART / ("fence-" + str(len(fences)+1) + ".sh")
        p.write_bytes(raw)
        fences.append({"path": p.name, "sha256": hashlib.sha256(raw).hexdigest()})
        run(["bash", "-n", str(p)], p.stem + "-syntax")
    elif opened:
        fence.append(line)
assert len(fences) == 4
run([sys.executable, "scripts/fetch_natural_release.py", "--manifest",
     "docs/natural-assistant/v4/public-release.json", "--list"], "release-list")
listed = json.loads((ART / "release-list.stdout.txt").read_bytes())
assert listed["selected_variant"] == "base"
assert manifest["adapter_parameters"] == 0
assert {x["output"] for x in manifest["files"]} == {"README.md", "release-provenance.json"}
assert release.sha256(ROOT / "requirements-natural.txt") == manifest["requirements_sha256"]
assert (ART / "official/public-pinned-requirements.txt").read_bytes() == (ROOT / "requirements-natural.txt").read_bytes()

# Fetch only the two metadata files allowed by the original manifest. Never read
# the custom release card's editorial content; the original verifier checks bytes.
public_dir = ART / "public-download"
metadata_env = dict(os.environ, HF_HOME=str(ART / "metadata-only-hf-cache"),
                         HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
run([sys.executable, "scripts/fetch_natural_release.py", "--manifest",
     "docs/natural-assistant/v4/public-release.json", "--output", str(public_dir)],
    "release-fetch", env=metadata_env)
run([sys.executable, "scripts/fetch_natural_release.py", "--manifest",
     "docs/natural-assistant/v4/public-release.json", "--output", str(public_dir), "--verify"],
    "release-verify", env=metadata_env)
release.verify_release(manifest, public_dir)
public_file_hashes = {p.name: release.sha256(p) for p in public_dir.iterdir() if p.is_file()}

options = release.student_options(manifest, public_dir, device="cpu", dtype="float32")
assert options.adapter is None and options.dtype == "float32" and options.device == "cpu"
for dtype in ("float16", "bfloat16"):
    try:
        release.student_options(manifest, public_dir, device="cpu", dtype=dtype)
    except ValueError:
        pass
    else:
        raise AssertionError("CPU non-float32 unexpectedly accepted")
try:
    release.check_runtime(manifest)
except ValueError as error:
    actual_runtime_rejection = str(error)
else:
    raise AssertionError("The current Python 3.13 environment must not pass the Python 3.12 gate")
with patch.object(release.sys, "version_info", (3, 12, 0)), patch.object(
    release.importlib.metadata, "version", side_effect=lambda name: manifest["dependency_versions"][name]
):
    release.check_runtime(manifest)

# Loader argument forwarding: intentionally tiny stand-ins, zero weights/inference.
calls = []
class FakeModel:
    def to(self, device):
        calls.append(("to", device)); return self
    def requires_grad_(self, value):
        calls.append(("requires_grad", value)); return self
    def eval(self):
        calls.append(("eval",)); return self
def loader(name, model=False):
    return types.SimpleNamespace(from_pretrained=lambda repo, **kw: (
        calls.append((name, repo, {k: str(v) for k,v in kw.items()})) or (FakeModel() if model else object())))
fake_transformers = types.ModuleType("transformers")
for name, model in [("AutoProcessor", False), ("Qwen3VLForConditionalGeneration", True),
                    ("WhisperProcessor", False), ("WhisperForConditionalGeneration", True)]:
    setattr(fake_transformers, name, loader(name, model))
with patch.dict(sys.modules, {"transformers": fake_transformers}):
    core.load_core(options, adapter=options.adapter)
    core.load_asr(options)
qwen_calls = [x for x in calls if x[0] == "Qwen3VLForConditionalGeneration"]
asr_calls = [x for x in calls if x[0] == "WhisperForConditionalGeneration"]
assert qwen_calls[0][2]["revision"] == manifest["base_model"]["revision"]
assert asr_calls[0][2]["revision"] == manifest["asr_model"]["revision"]
assert qwen_calls[0][2]["dtype"] == "torch.float32"
assert qwen_calls[0][2]["attn_implementation"] == "sdpa"
assert ("requires_grad", False) in calls

# UI data-flow test uses real server operations with stand-ins for model output.
rows, asr_loads = [], []
def fake_generate(model, processor, row, root, opts):
    rows.append(copy.deepcopy(row))
    return {"prediction": "bounded CPU contract stand-in"}
def fake_asr(opts):
    asr_loads.append(True); return object(), object()
def fake_transcribe(model, processor, path):
    return {"transcript": "辨識原文字"}
with patch.object(core, "generate", fake_generate), patch.object(core, "load_asr", fake_asr), patch.object(core, "transcribe", fake_transcribe):
    server = ui.create_server(object(), object(), options, ART / "bounded-ui-data", port=0)
    try:
        session_id = server.operation("/api/session", {})["session"]
        session = server.sessions[session_id]
        server.operation("/api/chat", {"session": session_id, "prompt": "你好"})
        assert not asr_loads
        image_bytes = io.BytesIO()
        Image.new("RGB", (2,2), (10,20,30)).save(image_bytes, format="PNG")
        image = server.operation("/api/upload", {"session": session_id, "kind": "image", "filename": "tiny.png",
                  "base64": base64.b64encode(image_bytes.getvalue()).decode()})["asset"]
        server.operation("/api/chat", {"session": session_id, "prompt": "圖中有哪些東西", "image": image})
        assert not asr_loads
        # Set a bounded audio fixture directly; transcription is an explicit stand-in.
        audio_path = Path(server.upload_directory.name) / "bounded-audio.wav"
        audio_path.write_bytes(b"bounded-no-model-fixture")
        session["assets"]["audio"] = {"path": str(audio_path.relative_to(server.data_root)), "kind": "audio", "media_type": "audio/wav"}
        transcription = server.operation("/api/transcribe", {"session": session_id, "audio": "audio"})
        assert len(rows) == 2 and len(asr_loads) == 1
        assert transcription["asr"]["transcript"] == "辨識原文字"
        result = server.operation("/api/chat", {"session": session_id, "prompt": "人工更正後文字", "speech": "audio"})
        assert result["user"] == "人工更正後文字" and result["asr"]["corrected"]
        assert rows[-1]["user"] == "人工更正後文字"
        assert any(part["type"] == "image" for turn in rows[-1]["history"] for part in turn["content"])
        assert any(part["type"] == "image" for turn in core.messages_for(rows[-1], server.data_root) for part in turn["content"])
        before_reset = [core.asset_path(a["path"],server.data_root) for a in session["assets"].values()]
        server.operation("/api/reset", {"session": session_id})
        assert not session["history"] and not session["assets"] and not session["transcriptions"]
        assert all(not p.exists() for p in before_reset)
        server_address = server.server_address
    finally:
        server.server_close()
    assert not Path(server.upload_directory.name).exists()
assert all(label in ui.PAGE for label in ["要送出的文字", "送出問題", "辨識語音", "辨識原稿", "開始新對話"])

# Independent recomputation of existing raw measurements, with no grading replay.
original_base = ART / "originals/docs/natural-assistant/evidence/v4-research/student-selected-public-ui"
raw_report = json.loads((original_base / "actual-ui/report.json").read_bytes())
events = [json.loads(x) for x in (original_base / "actual-ui/observer.jsonl").read_bytes().splitlines()]
assert events[0]["inference_mock"] is False
assert raw_report["git_revision"] == "59a1eda4ed7b6e8609892ec2b9013c821ac93e69"
assert raw_report["process_exit_code"] == 0 and raw_report["timed_out"] is False
for name, expected in manifest["dependency_versions"].items():
    assert raw_report["dependencies"][name].split("+",1)[0] == expected
for src in raw_report["sources_before"]:
    if src["path"] in ["scripts/fetch_natural_release.py", "scripts/natural_assistant.py", "tiny_perceptron/natural_assistant.py", "tiny_perceptron/natural_ui.py", "requirements-natural.txt"]:
        assert release.sha256(ART / "pinned" / src["path"]) == src["sha256"]
loads = {name: sum(e["kind"] == name + "_begin" for e in events) for name in ["load_core", "load_asr"]}
assert loads == {"load_core": 1, "load_asr": 1}
generations = [e["result"] for e in events if e["kind"] == "generate_end"]
transcriptions = [e["result"] for e in events if e["kind"] == "transcribe_end"]
assert len(generations) == 4 and len(transcriptions) == 1
for result in generations:
    token_ids = result["generated_token_ids"]
    assert len(token_ids) == result["generated_tokens"]
    ended = bool(token_ids and token_ids[-1] in result["eos_token_ids"])
    assert ended == result["ended_with_eos"]
    assert result["stop_reason"] == ("eos" if ended else "max_new_tokens" if len(token_ids) >= manifest["runtime"]["max_new_tokens"] else "unknown")
    assert isinstance(result["prediction"], str) and result["prediction"].strip()
audio = transcriptions[0]
assert len(audio["raw_token_ids"]) == audio["raw_token_count"]
prefix = audio["expected_decoder_prompt_ids"]
assert audio["raw_token_ids"][:len(prefix)] == prefix
assert audio["raw_token_ids"][len(prefix):] == audio["generated_token_ids"]
assert len(audio["generated_token_ids"]) == audio["generated_token_count"]
assert (audio["raw_token_ids"][-1] in audio["eos_token_ids"]) == audio["ended_with_eos"]
begins = [e for e in events if e["kind"] == "generate_begin"]
history_image_counts = [sum(p["type"] == "image" for t in e["row"].get("history",[]) for p in t["content"]) for e in begins]
assert history_image_counts[2] >= 1 and history_image_counts[3] >= 1
reset_states = [e["state"] for e in events if e["kind"] == "operation_end" and e["route"] == "/api/reset"]
assert len(reset_states) == 1
assert all(reset_states[0][k] == 0 for k in ["history_messages", "assets", "transcriptions"])
disk = json.loads((original_base / "actual-ui/disk-observation.json").read_bytes())
disk_bytes = disk["official_cache"]["unique_regular_file_bytes"]
assert disk_bytes > 3 * 1024**3

recomputed = {
    "raw_pointers_read": {
        "report.json": ["/git_revision", "/command", "/sources_before", "/dependencies", "/torch_threads", "/torch_interop_threads", "/process_exit_code", "/timed_out"],
        "observer.jsonl": ["/0/inference_mock", "/0/sources", "load_core_begin/options", "load_asr_begin/options", "generate_begin/row/history", "generate_end/result/{prediction,generated_tokens,generated_token_ids,eos_token_ids,ended_with_eos,stop_reason}", "transcribe_end/result/{raw_token_ids,raw_token_count,expected_decoder_prompt_ids,generated_token_ids,generated_token_count,eos_token_ids,ended_with_eos}", "operation_end/{route,state}"],
        "disk-observation.json": ["/official_cache/unique_regular_file_bytes"],
    },
    "existing_load_counts": loads,
    "existing_chat_turns": len(generations), "existing_transcriptions": len(transcriptions),
    "existing_generated_token_counts": [len(r["generated_token_ids"]) for r in generations],
    "existing_generation_end_is_eos": [r["generated_token_ids"][-1] in r["eos_token_ids"] for r in generations],
    "existing_asr_generated_token_count": len(audio["generated_token_ids"]),
    "existing_history_image_counts": history_image_counts,
    "existing_reset_state": reset_states[0], "existing_official_cache_bytes": disk_bytes,
    "quality_scope": "One existing Linux CPU trial, four chat turns and one transcription. Connectivity/state only; no held-out accuracy, general reliability or new model evaluation.",
}
results = {
    "bash_fences": fences, "commands": commands, "public_file_hashes": public_file_hashes,
    "actual_runtime_rejection": actual_runtime_rejection,
    "runtime_gate_expected_version_test": "Mocked version metadata only; real requirement and code SHA checks",
    "loader_standin_calls": calls, "ui_standin_generated_calls": len(rows),
    "ui_standin_asr_loads": len(asr_loads), "ui_loopback_address": list(server_address),
    "recomputed_existing_evidence": recomputed,
    "new_full_model_runs": 0, "new_model_weight_downloads": 0, "new_training_runs": 0,
}
(ART / "verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2))
