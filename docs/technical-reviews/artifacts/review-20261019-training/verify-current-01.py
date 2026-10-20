"""以隔離的短程算例複查裝置介面與答案比較；既有證據唯讀。"""

import ast
import contextlib
import difflib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
WORK = BASE / "work-current-01"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
import torch
from scripts.course_experiments import run as experiment_run
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.training import choose_device

torch.set_num_threads(2)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
REPORT = Path("docs/course-review-20261019/reviews/rv1019_technical_training/technical-training-original-02.json")
OLD = Path("docs/course-review-20261019/freeze-01/sources/training.md")
NEW = Path("docs/course-review-20261019/freeze-06/sources/training.md")
previous = json.loads(REPORT.read_bytes())
out = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-training/verify-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu", "cuda_available": str(torch.cuda.is_available()), "mps_available": str(torch.backends.mps.is_available())},
    "original_report": {"path": str(REPORT), "sha256": sha(REPORT)},
    "freeze": {"path": str(NEW), "sha256": sha(NEW), "manifest_path": "docs/course-review-20261019/freeze-06.json", "manifest_sha256": sha("docs/course-review-20261019/freeze-06.json")},
    "reuse_audit": {}, "changes": [], "runs": [], "observations": {},
}
assert out["original_report"]["sha256"] == "d644480ecc2628587c19432a31d89f68228be66399ae4ef87c1ca6b5e8656c10"
assert out["freeze"]["sha256"] == "d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
assert sha("course/training.md") == out["freeze"]["sha256"]
for group in ("sources", "artifacts"):
    entries = []
    for item in previous[group]:
        if "path" in item and "sha256" in item:
            entries.append({"id": item["id"], "path": item["path"], "expected_sha256": item["sha256"], "current_sha256": sha(item["path"])})
    assert all(item["expected_sha256"] == item["current_sha256"] for item in entries)
    out["reuse_audit"][group] = entries
old_lines, new_lines = OLD.read_text().splitlines(), NEW.read_text().splitlines()
for tag, a, b, c, d in difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes():
    if tag != "equal":
        out["changes"].append({"operation": tag, "old_line_start": a + 1, "new_line_start": c + 1, "before": old_lines[a:b], "after": new_lines[c:d]})
assert len(out["changes"]) == 3
out["observations"]["figures"] = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", NEW.read_text())
assert not out["observations"]["figures"]


def run(arguments, expected=0):
    command = [sys.executable, *map(str, arguments)]
    completed = subprocess.run(command, cwd=ROOT, env=os.environ, capture_output=True, text=True, timeout=45)
    item = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    out["runs"].append(item)
    assert completed.returncode == expected, item
    return completed.stdout


class Plain(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, value):
        if not self.skip:
            self.text.append(value)


WORK.mkdir(exist_ok=False)
try:
    out["observations"]["device_priority"] = []
    for cuda, mps in ((True, True), (False, True), (False, False)):
        with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
            out["observations"]["device_priority"].append({"cuda_available": cuda, "mps_available": mps, "auto": str(choose_device()), "explicit_cpu": str(choose_device("cpu"))})
    capture = {}

    def intercepted(experiment_id, device, output, dependencies, assets, **kwargs):
        capture.update(experiment_id=experiment_id, device=device, output=str(output), dependencies=str(dependencies), scope="僅執行真實CLI解析；攔截execute，不執行固定實驗")
        return {"experiment_id": experiment_id, "elapsed_seconds": 0, "evidence_status": "parser_intercept", "artifacts": []}

    with patch.object(sys, "argv", ["scripts.course_experiments.run", "--experiment", "sft", "--output", str(WORK / "experiments")]), patch.object(experiment_run, "execute", intercepted), contextlib.redirect_stdout(io.StringIO()) as stdout:
        experiment_run.main()
    out["observations"]["experiment_default"] = {**capture, "stdout": stdout.getvalue()}
    assert capture["device"] == "cpu"
    assert not (WORK / "experiments").exists()
    run(["scripts/train.py", "--task", "text", "--output", WORK / "unused.pt"])
    run(["scripts/train.py", "--task", "vision", "--output", WORK / "unused-vision.pt"])
    assert not (WORK / "unused.pt").exists()
    assert not (WORK / "unused-vision.pt").exists()

    model = TinyLM(ModelConfig(width=8, layers=1, heads=1, max_length=128))
    tok = ByteTokenizer()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.final_norm.bias[0] = 1
        model.output.weight[tok.encode("x")[0], 0] = 7
    payload = {"format_version": 1, "config": asdict(model.config), "model": model.state_dict(), "optimizer": None, "step": 0, "tokenizer": {"type": "byte", "vocab_size": tok.vocab_size}, "metadata": {"purpose": "人工建構的答案一致但不正確反例；未訓練模型"}}
    memory = io.BytesIO()
    torch.save(payload, memory)
    native = WORK / "native.pt"
    with native.open("xb") as stream:
        stream.write(memory.getvalue())
    packed = WORK / "packed4.pt"
    prompt = "color=blue;shape=circle;pitch=low;color?"
    common = ["--chat", "--prompt", prompt, "--tokens", "24", "--json"]
    before = json.loads(run(["scripts/infer.py", native, *common]))
    before_cpu = json.loads(run(["scripts/infer.py", native, *common, "--device", "cpu"]))
    assert before == before_cpu
    run(["scripts/quantize.py", native, "--bits", "4", "--output", packed])
    after = json.loads(run(["scripts/infer.py", packed, *common]))
    after_cpu = json.loads(run(["scripts/infer.py", packed, *common, "--device", "cpu"]))
    assert after == after_cpu
    assert before["answer"] == after["answer"] == "x" * 24
    assert before["answer"] != "blue"
    state = torch.load(packed, map_location="cpu", weights_only=True)["model"]
    out["observations"]["quantization_counterexample"] = {"configuration": payload["config"], "construction": "全參數先設0；final_norm.bias[0]=1；output.weight[byte(x),0]=7；沒有訓練。", "prompt": prompt, "target": "blue", "before": before, "after": after, "answers_equal": before["answer"] == after["answer"], "before_correct": before["answer"] == "blue", "after_correct": after["answer"] == "blue", "saved_devices": sorted({str(t.device) for t in state.values()}), "scope": "透明人工反例，只證明答案一致不能推出答對，不測量500步配方的成功率或量化的普遍品質。"}
    row = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": "blue"}]}
    data = WORK / "fixture.jsonl"
    with data.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    evaluation = WORK / "evaluation.json"
    run(["scripts/evaluate.py", native, "--data", data, "--mode", "sft", "--tokens", "24", "--limit", "all", "--output", evaluation])
    out["observations"]["evaluation_fixture"] = json.loads(evaluation.read_bytes())
    for arguments in (
        ["scripts/quantize.py", native, "--device", "cpu", "--output", WORK / "not-created.pt"],
        ["scripts/prepare_data.py", "--device", "cpu", "--output", WORK / "not-created-data"],
        ["scripts/fetch_training_assets.py", "--list", "--device", "cpu"],
    ):
        run(arguments, expected=2)
    assert not (WORK / "not-created.pt").exists()
    assert not (WORK / "not-created-data").exists()

    url = "https://docs.pytorch.org/docs/2.14/notes/randomness.html"
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    fetched = subprocess.run(command, capture_output=True, check=True)
    plain = Plain()
    plain.feed(fetched.stdout.decode())
    text = re.sub(r"\s+", " ", " ".join(plain.text))
    phrase = "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
    assert phrase in text and "CPU and GPU executions, even when using identical seeds" in text
    out["reproducibility_source"] = {"url": url, "version": "PyTorch 2.14", "accessed_on": "2026-10-21", "command": command, "returncode": fetched.returncode, "original_sha256": hashlib.sha256(fetched.stdout).hexdigest(), "locator": "Reproducibility：開頭兩段；Controlling sources of randomness", "excerpt": phrase, "inspection_note": "親讀官方原文：跨版本、commit及平台不保證完全重現，即使相同seed，CPU/GPU也可能不同。此來源支持可能性，不支持所有硬體必然不同或小模型短訓必然失敗。"}
    snapshots = {}
    for filename, identifiers in {
        "tiny_perceptron/training.py": ("choose_device", "seed_everything", "load_checkpoint"),
        "scripts/train.py": ("parser",),
        "scripts/prepare_data.py": ("main",),
        "scripts/fetch_training_assets.py": ("main",),
        "scripts/quantize.py": ("main",),
        "scripts/course_experiments/run.py": ("main",),
    }.items():
        text = Path(filename).read_text()
        lines = text.splitlines()
        snapshots[filename] = {"sha256": sha(filename), "identifiers": {node.name: "\n".join(lines[node.lineno - 1:node.end_lineno]) for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in identifiers}}
    out["source_snapshots"] = snapshots
    out["status"] = "completed"
finally:
    shutil.rmtree(WORK)
    with (BASE / "verification-current-01.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": out["status"], "runs": len(out["runs"]), "changed_blocks": len(out["changes"]), "unchanged_repository_sources": len(out["reuse_audit"]["sources"]), "unchanged_artifacts": len(out["reuse_audit"]["artifacts"]), "counterexample": {k: out["observations"]["quantization_counterexample"][k] for k in ("answers_equal", "before_correct", "after_correct")}}, ensure_ascii=False))
