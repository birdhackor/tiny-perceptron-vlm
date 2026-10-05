"""Bounded, CPU-only verification of section 12.14; no model or data download."""
import ast
import contextlib
import functools
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tiny_perceptron.natural_concepts import speech_stages, text_error_report
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.lru_cache(None)
def independent_distance(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    return min(
        independent_distance(a[1:], b) + 1,
        independent_distance(a, b[1:]) + 1,
        independent_distance(a[1:], b[1:]) + (a[0] != b[0]),
    )


section = (OUT / "inputs/section12_14.md").read_bytes()
fences = re.findall(rb"```python\n(.*?)```", section, re.S)
assert len(fences) == 1
(OUT / "fence-1.py").write_bytes(fences[0])
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(fences[0], "course/chapters/12.md#12.14:fence-1", "exec"), {})
assert stdout.getvalue() == "聽寫字元錯誤率 0.25\n兩路回答完全相同 True\n相同是否足以證明正確 None\n"
print("ORIGINAL FENCE STDOUT")
print(stdout.getvalue(), end="")

cases = [
    ("我不吃辣", "我不吃拉", 1, 4, 0.25),
    ("我不吃辣", "我不吃辣", 0, 4, 0.0),
    ("不要加辣", "要加辣", 1, 4, 0.25),
    ("我不吃辣", "我真的不吃辣", 2, 4, 0.5),
    ("", "辣", 1, 0, None),
]
measured = []
for reference, recognized, edits, denominator, cer in cases:
    report = text_error_report(reference, recognized)
    assert report["edits"] == edits == independent_distance(reference, recognized)
    assert report["reference_characters"] == denominator
    assert report["cer"] == cer
    measured.append(report)
print("EDIT CASES", json.dumps(measured, ensure_ascii=False))
agreement = speech_stages("我不吃辣", "我不吃辣", "加辣火鍋", "加辣火鍋")
disagreement = speech_stages("我不吃辣", "我不吃辣", "清淡湯麵", "加辣火鍋")
assert agreement["answers_identical"] is True and agreement["answers_correct"] is None
assert disagreement["answers_identical"] is False and disagreement["answers_correct"] is None
print("ANSWER CASES", json.dumps([agreement, disagreement], ensure_ascii=False))

# Compile the original UI operation/session methods alone. Generation is a stub:
# this exercises data routing and history retention, not ASR or model quality.
ui_path = ROOT / "tiny_perceptron/natural_ui.py"
ui_tree = ast.parse(ui_path.read_text())
server = next(n for n in ui_tree.body if isinstance(n, ast.ClassDef) and n.name == "NaturalServer")
methods = [n for n in server.body if isinstance(n, ast.FunctionDef) and n.name in {"session", "operation"}]
calls = []


def fake_generate(model, processor, row, data_root, options):
    calls.append(json.loads(json.dumps(row)))
    return {"prediction": "清淡湯麵"}


namespace = {
    "assistant": SimpleNamespace(generate=fake_generate),
    "MAX_PROMPT_BYTES": 8192,
    "MAX_TURNS": 16,
}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(ui_path), "exec"), namespace)
BoundedServer = type("BoundedServer", (), {n.name: namespace[n.name] for n in methods})
bounded = BoundedServer()
bounded.sessions = {"test": {"history": [], "assets": {}, "transcriptions": {"speech": {"transcript": "還有別的嗎"}}}}
bounded.model = bounded.processor = bounded.options = None
bounded.data_root = ROOT
first = bounded.operation("/api/chat", {"session": "test", "prompt": "我不吃辣"})
second = bounded.operation("/api/chat", {"session": "test", "prompt": "還有別的嗎", "speech": "speech"})
assert calls[0]["history"] == []
assert calls[1]["history"][0]["content"] == [{"type": "text", "text": "我不吃辣"}]
assert len(bounded.sessions["test"]["history"]) == 4
assert second["asr"]["transcript"] == "還有別的嗎"
assert second["asr"]["submitted_text"] == "還有別的嗎"
assert second["asr"]["corrected"] is False
print("UI HISTORY CONTRACT", json.dumps({"calls": calls, "history": bounded.sessions["test"]["history"]}, ensure_ascii=False))

# Read only the named raw configuration pointers; no release review fields printed.
manifest_path = ROOT / "docs/natural-assistant/v4/public-release.json"
manifest = json.loads(manifest_path.read_text())
keys = ["release_id", "selected_variant", "base_model", "asr_model", "adapter_parameters"]
configuration = {k: manifest[k] for k in keys}
assert configuration["selected_variant"] == "base"
assert configuration["adapter_parameters"] == 0
fetch_path = ROOT / "scripts/fetch_natural_release.py"
tree = ast.parse(fetch_path.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "student_options")
namespace = {"validate_manifest": lambda m: None, "SimpleNamespace": SimpleNamespace, "Path": Path, "ROOT": ROOT}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(fetch_path), "exec"), namespace)
options = namespace["student_options"](manifest, OUT, device="cpu")
assert options.adapter is None
assert options.model == manifest["base_model"]["repo"]
print("BASE CONFIGURATION", json.dumps(configuration, ensure_ascii=False))
print("OPTIONS CONTRACT", json.dumps({"adapter": options.adapter, "device": options.device, "dtype": options.dtype}))
print("BOUNDED NOTE: student_options uses original AST with manifest validation stubbed; only option mapping is tested.")

environment = {
    "python": platform.python_version(), "python_executable": sys.executable,
    "torch": torch.__version__, "device": "cpu", "gpu_used": False,
    "model_loaded": False, "training_run": False,
    "source_sha256": hashlib.sha256(section).hexdigest(),
    "code_hashes": {str(p.relative_to(ROOT)): digest(p) for p in [
        ROOT / "tiny_perceptron/natural_concepts.py", ui_path,
        ROOT / "tiny_perceptron/natural_assistant.py", fetch_path, manifest_path,
    ]},
}
(OUT / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
print("ENVIRONMENT", json.dumps(environment, ensure_ascii=False))
print("ALL ASSERTIONS PASSED")
