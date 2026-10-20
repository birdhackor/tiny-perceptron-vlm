"""唯讀沿用既有證據，短程檢查線上合成題及固定模態對照。"""

import ast
import contextlib
import datetime
import difflib
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
WORK = BASE / "work-current-02"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
import torch
from scripts import train
from scripts.course_experiments import modalities
from scripts.course_experiments.common import evaluate_lm
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import AudioEncoder, VisionEncoder
from tiny_perceptron.training import load_checkpoint, seed_everything

torch.set_num_threads(2)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
content_hash = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
previous_path = Path("docs/course-review-20261019/reviews/rv1019_technical_training/technical-training-current-01.json")
old_source = Path("docs/course-review-20261019/freeze-06/sources/training.md")
new_source = Path("docs/course-review-20261019/freeze-07/sources/training.md")
previous = json.loads(previous_path.read_bytes())
out = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-training/verify-current-02.py", "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(), "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu"}, "previous_report": {"path": str(previous_path), "sha256": sha(previous_path)}, "freeze": {"path": str(new_source), "sha256": sha(new_source), "manifest": "docs/course-review-20261019/freeze-07.json", "manifest_sha256": sha("docs/course-review-20261019/freeze-07.json")}, "reuse_audit": {}, "changed_blocks": [], "runs": [], "source_snapshots": {}}
assert out["previous_report"]["sha256"] == "101dde958dd6c417d75dfb283cc5db53fb8ab3936906d90226da17b9c8921418"
assert out["freeze"]["sha256"] == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
assert sha("course/training.md") == out["freeze"]["sha256"]
for group in ("sources", "artifacts"):
    checked = [{"id": item["id"], "path": item["path"], "expected_sha256": item["sha256"], "current_sha256": sha(item["path"])} for item in previous[group] if "path" in item and "sha256" in item]
    assert all(row["expected_sha256"] == row["current_sha256"] for row in checked)
    out["reuse_audit"][group] = checked
old_lines, new_lines = old_source.read_text().splitlines(), new_source.read_text().splitlines()
for tag, a, b, c, d in difflib.SequenceMatcher(a=old_lines, b=new_lines).get_opcodes():
    if tag != "equal":
        out["changed_blocks"].append({"operation": tag, "old_start": a + 1, "new_start": c + 1, "before": old_lines[a:b], "after": new_lines[c:d]})
assert len(out["changed_blocks"]) == 2
assert "![" not in new_source.read_text()


def save_new(path, payload):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    with Path(path).open("xb") as stream:
        stream.write(buffer.getvalue())


def identity_rows(samples):
    return [{"messages": row["messages"], "expected": row["expected"]} for row in samples]


WORK.mkdir(exist_ok=False)
try:
    seed_everything(42)
    config = ModelConfig(width=32, layers=1, heads=1, max_length=128)
    baseline = TinyLM(config)
    native = WORK / "parent.pt"
    save_new(native, {"format_version": 1, "config": asdict(config), "model": baseline.state_dict(), "optimizer": None, "step": 0, "tokenizer": {"type": "byte", "vocab_size": config.vocab_size}, "metadata": {"purpose": "隨機初始化的通路測試起點，不是完成訓練的教學模型"}})
    vision_encoder, audio_encoder = WORK / "vision-encoder.pt", WORK / "audio-encoder.pt"
    save_new(vision_encoder, {"encoder": VisionEncoder().state_dict()})
    save_new(audio_encoder, {"encoder": AudioEncoder().state_dict()})
    text_rows = [
        {"messages": [{"role": "user", "content": "color=blue;shape=circle;pitch=low;color?"}, {"role": "assistant", "content": "blue"}]},
        {"messages": [{"role": "user", "content": "color=red;shape=square;pitch=high;shape?"}, {"role": "assistant", "content": "square"}]},
    ]
    original_choice = random.choice
    for task, freeze in (("vision", "projector"), ("audio", "projector"), ("joint", "partial")):
        checkpoint = WORK / (task + ".pt")
        arguments = ["scripts/train.py", "--task", task, "--checkpoint", str(native), "--freeze", freeze, "--train", "--steps", "500", "--stop-after", "2", "--device", "cpu", "--output", str(checkpoint)]
        if task != "audio":
            arguments += ["--vision-encoder", str(vision_encoder)]
        if task != "vision":
            arguments += ["--audio-encoder", str(audio_encoder)]
        choices = []

        def traced_choice(values):
            selected = original_choice(values)
            choices.append({"population": list(values), "selected": selected})
            return selected

        before = evaluate_lm(baseline, text_rows, mode="sft", tokens=8)
        with patch.object(sys, "argv", arguments), patch.object(train.random, "choice", traced_choice), contextlib.redirect_stdout(io.StringIO()) as stream:
            train.main()
        stdout = stream.getvalue()
        report = json.loads(stdout.splitlines()[-1])
        loaded, payload = load_checkpoint(checkpoint, "cpu")
        assert report["mode"] == "train" and len(report["history"]) == 2
        assert payload["step"] == 2 and payload["metadata"]["schedule_steps"] == 500
        assert payload["metadata"]["data"] == "None"
        assert len(choices) == 3 * 4 * 2
        draws = []
        for start in range(0, len(choices), 3):
            colour, shape, high = choices[start:start + 3]
            assert colour["population"] == ["red", "green", "blue"]
            assert shape["population"] == ["circle", "square"]
            assert high["population"] == [False, True]
            draws.append({"step": start // (3 * 4) + 1, "colour": colour["selected"], "shape": shape["selected"], "high": high["selected"]})
        after = evaluate_lm(loaded.language, text_rows, mode="sft", tokens=8)
        assert before["records"] == after["records"] == len(text_rows)
        assert identity_rows(before["samples"]) == identity_rows(after["samples"])
        assert all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in report["history"])
        out["runs"].append({"command": [sys.executable, *arguments], "invocation": "以真實train.main及CLI parser執行；只攔截random.choice記錄輸入，實際選樣仍呼叫原函式", "stdout": stdout, "returncode": 0, "task": task, "generated_examples": draws, "recorded_random_draws": len(choices), "report_keys": list(report), "payload_keys": list(payload), "training_metadata_keys": list(payload["metadata"]), "configuration": payload["config"], "step": payload["step"], "schedule_steps": payload["metadata"]["schedule_steps"], "text_fixture_input_sha256": content_hash(text_rows), "text_fixture_before": before, "text_fixture_after": after, "scope": "只跑原500步排程的前2步；起點及encoder均未訓練。另以兩筆人工文字fixture示範model.language成對評估，不作原教學模型的能力或退步結論。"})
    command = [sys.executable, "scripts/evaluate.py", str(WORK / "joint.pt"), "--mode", "text", "--device", "cpu", "--output", str(WORK / "not-created.json")]
    rejected = subprocess.run(command, cwd=ROOT, env=os.environ, capture_output=True, text=True, timeout=30)
    assert rejected.returncode == 1 and "TinyLM" in rejected.stderr
    assert not (WORK / "not-created.json").exists()
    out["runs"].append({"command": command, "returncode": rejected.returncode, "stdout": rejected.stdout, "stderr": rejected.stderr})

    vqa_path = Path("docs/course-experiments/results/vqa.json")
    vqa = json.loads(vqa_path.read_bytes())["results"]
    generated = modalities._vision_records(("shape?", "color?"))
    expected = {name: vqa["data"]["splits"][name]["records"] for name in ("train", "validation", "test")}
    assert generated == expected
    fixed = {"path": str(vqa_path), "sha256": sha(vqa_path), "modal_split_counts": {name: len(rows) for name, rows in expected.items()}, "modal_splits_match_current_generator": generated == expected, "text_data": vqa["text_data"], "text_pairs": {}, "scope": "直接核對保存的原始GPU結果及現行配方，不重新訓練GPU實驗。"}
    for name, branch in vqa["variants"].items():
        if "text_before" not in branch:
            fixed["text_pairs"][name] = {"paired_text_comparison_present": False, "scope": "此direct_vqa分支沒有成對text_before/text_after，不將其他分支的檢查外推至此分支。"}
            continue
        first, second = branch["text_before"], branch["text_after"]
        before_identity, after_identity = identity_rows(first["samples"]), identity_rows(second["samples"])
        assert before_identity == after_identity
        assert first["records"] == second["records"] == vqa["text_data"]["test"]["count"]
        assert len(first["samples"]) == len(second["samples"]) == first["records"]
        fixed["text_pairs"][name] = {"paired_text_comparison_present": True, "records_before": first["records"], "records_after": second["records"], "questions_and_targets_identical": True, "questions_and_targets_sha256": content_hash(before_identity), "text_probe_is_training_diagnostic": branch["text_probe_is_training_diagnostic"], "before_metric_keys": list(first), "after_metric_keys": list(second)}
    assert sum(item["paired_text_comparison_present"] for item in fixed["text_pairs"].values()) == 4
    out["fixed_vqa"] = fixed
    for filename, names in {
        "scripts/train.py": ("modal_loss", "prepare_examples", "main"),
        "scripts/evaluate.py": ("main", "evaluate"),
        "scripts/course_experiments/modalities.py": ("_vision_records", "_manifest", "run_projector", "run_vqa"),
        "scripts/course_experiments/common.py": ("evaluate_lm",),
    }.items():
        raw = Path(filename).read_text()
        lines = raw.splitlines()
        out["source_snapshots"][filename] = {"sha256": sha(filename), "identifiers": {node.name: "\n".join(lines[node.lineno - 1:node.end_lineno]) for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and node.name in names}}
    out["guide"] = {"path": "docs/technical-review-guide.md", "sha256": sha("docs/technical-review-guide.md")}
    out["status"] = "completed"
finally:
    shutil.rmtree(WORK)
    out["finished_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    with (BASE / "verification-current-02.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": out["status"], "changed_blocks": len(out["changed_blocks"]), "cli_runs": len(out["runs"]), "training_steps_each_route": 2, "examples_each_route": 8, "fixed_vqa_text_pairs": 4, "unchanged_sources": len(out["reuse_audit"]["sources"]), "unchanged_artifacts": len(out["reuse_audit"]["artifacts"])}, ensure_ascii=False))
