"""5.7 的有界 CPU 診斷；不啟動正式訓練或 GPU。"""

import contextlib
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch

from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss
from tiny_perceptron.multimodal import MultiModalLM
from tiny_perceptron.training import load_checkpoint, save_checkpoint


ROOT = Path(__file__).resolve().parents[4]
torch.set_num_threads(2)
snapshot = ROOT / "docs/course-review-20261019/freeze-01/sources/5.7.md"
source = snapshot.read_text()
embedded = re.search(r"```python\n(.*?)```", source, re.S).group(1)
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_5_7",
    "environment": {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "platform": platform.platform(),
        "device": "cpu",
        "threads": str(torch.get_num_threads()),
    },
    "source_snapshot": str(snapshot.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "embedded_sha256": hashlib.sha256(embedded.encode()).hexdigest(),
}

namespace = {}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(embedded, "5.7-embedded.py", "exec"), namespace)
result["embedded"] = {
    "stdout": output.getvalue(),
    "both_assertions_passed": True,
    "input_shape": list(namespace["x"].shape),
    "input_dtype": str(namespace["x"].dtype),
    "logits_shape": list(namespace["model"](namespace["x"])["logits"].shape),
    "step": namespace["payload"]["step"],
    "keys": sorted(namespace["payload"]),
    "optimizer_state_keys": sorted(next(iter(namespace["payload"]["optimizer"]["state"].values()))),
    "effective_targets": int((namespace["y"] != -100).sum()),
}

exercise = embedded.replace('    restored_optimizer.load_state_dict(payload["optimizer"])\n', '')
namespace = {}
try:
    exec(compile(exercise, "5.7-exercise.py", "exec"), namespace)
except AssertionError as error:
    frame = traceback.extract_tb(error.__traceback__)[-1]
    lines = exercise.splitlines()
    assert frame.lineno == [i + 1 for i, line in enumerate(lines) if "assert torch.equal" in line][-1]
    difference = (namespace["model"](namespace["x"])["logits"] - namespace["loaded"](namespace["x"])["logits"]).abs().max()
    result["exercise"] = {"first_assertion_passed": True, "second_assertion_failed": True, "failure_line": frame.lineno, "max_logit_difference": float(difference.detach())}
else:
    raise AssertionError("練習應在第二個 assert 失敗")

with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    torch.manual_seed(42)
    random.seed(42)
    model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    masked_loss(model(x)["logits"], y).backward()
    opt.step()
    model.embedding.weight.requires_grad_(False)
    path = directory / "state.pt"
    save_checkpoint(path, model, opt, step=1, metadata={"chars": ["。", "狗", "看", "貓", "，"]}, training_state={"progress": 1})
    expected_torch, expected_python = torch.rand(5), random.choices(range(10), k=8)
    torch.rand(11)
    random.random()
    loaded, payload = load_checkpoint(path, restore_rng=True)
    observed_torch, observed_python = torch.rand(5), random.choices(range(10), k=8)
    assert torch.equal(expected_torch, observed_torch) and expected_python == observed_python
    assert not loaded.embedding.weight.requires_grad
    assert payload["training_state"] == {"progress": 1}
    assert payload["tokenizer"] == {"type": "unspecified", "vocab_size": 5}
    assert torch.equal(model(x)["logits"], loaded(x)["logits"])
    result["rng_and_payload"] = {"torch_cpu_sequence_equal": True, "python_sampling_equal": True, "python_sequence": observed_python, "trainable_mask_restored": True, "training_state": payload["training_state"], "tokenizer": payload["tokenizer"], "config": payload["config"], "adam_step_values": sorted({float(s["step"]) for s in payload["optimizer"]["state"].values()})}

    inference_path = directory / "inference.pt"
    save_checkpoint(inference_path, model)
    inference, inference_payload = load_checkpoint(inference_path)
    assert inference_payload["optimizer"] is None
    assert torch.equal(model(x)["logits"], inference(x)["logits"])
    result["weights_without_optimizer"] = {"prediction_equal": True, "optimizer": None}

    original = path.read_bytes()
    def interrupted_save(value, target):
        Path(target).write_bytes(b"partial diagnostic write")
        raise RuntimeError("simulated interruption before replace")
    with patch("tiny_perceptron.training.torch.save", side_effect=interrupted_save):
        try:
            save_checkpoint(path, model)
        except RuntimeError:
            pass
    assert path.read_bytes() == original
    assert path.with_name("state.pt.tmp").read_bytes() == b"partial diagnostic write"
    result["interrupted_write"] = {"previous_checkpoint_unchanged": True, "partial_tmp_only": True, "power_loss_or_concurrent_writers_tested": False}

    for name, mutation, exception in [
        ("unknown_format", lambda p: p.update(format_version=999), ValueError),
        ("missing_weight", lambda p: p["model"].pop(next(iter(p["model"]))), RuntimeError),
    ]:
        candidate = torch.load(path, weights_only=True)
        mutation(candidate)
        altered = directory / (name + ".pt")
        torch.save(candidate, altered)
        try:
            load_checkpoint(altered)
        except exception as error:
            result[name] = {"rejected": True, "exception": type(error).__name__, "message": str(error)[:300]}
        else:
            raise AssertionError(name)

    modal = MultiModalLM(TinyLM(ModelConfig(width=8, max_length=8)))
    modal_path = directory / "modal.pt"
    save_checkpoint(modal_path, modal, metadata={"task": "joint"})
    restored_modal, modal_payload = load_checkpoint(modal_path)
    assert isinstance(restored_modal, MultiModalLM)
    assert modal_payload["task"] == "joint" and modal_payload["format_version"] == "multimodal-v1"
    result["multimodal_payload"] = {"modal_config": modal_payload["modal_config"], "task": modal_payload["task"], "format_version": modal_payload["format_version"], "strict_state_restore_passed": True}

    cli_model = TinyLM(ModelConfig(width=8, max_length=8))
    cli_opt = torch.optim.AdamW(cli_model.parameters(), lr=0.01)
    metadata = {"task": "text", "seed": 42, "data": "None", "data_sha256": None, "schedule_steps": 2, "peak_lr": 0.01, "batch_size": 1, "freeze": "none"}
    cli_path = directory / "cli.pt"
    save_checkpoint(cli_path, cli_model, cli_opt, step=1, metadata=metadata)
    base_command = [sys.executable, "scripts/train.py", "--task", "text", "--checkpoint", str(cli_path), "--resume", "--device", "cpu", "--batch-size", "1", "--steps", "2", "--lr", "0.01"]
    result["cli_diagnostics"] = []
    for name, command in [("matching_schedule_dry_run", base_command), ("changed_total", base_command + ["--steps", "3"]), ("changed_peak", base_command + ["--lr", "0.02"])]:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert proc.returncode == (0 if name == "matching_schedule_dry_run" else 1)
        if proc.returncode:
            assert "resume 的" in proc.stderr and "與原訓練不同" in proc.stderr
        else:
            report = json.loads(proc.stdout.strip().splitlines()[-1])
            assert report["mode"] == "dry-run-no-weight-update"
        result["cli_diagnostics"].append({"case": name, "command": command, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})

record_path = ROOT / "docs/course-experiments/results/text_foundation.json"
record = json.loads(record_path.read_text())
related = ["tiny_perceptron/training.py", "scripts/course_experiments/text.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py"]
result["historical_gpu_record_audit"] = {
    "path": str(record_path.relative_to(ROOT)),
    "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
    "recorded_environment": {key: record[key] for key in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status", "status"]},
    "resume": record["results"]["resume"],
    "historical_vs_current_code_sha256": {path: {"recorded": record["code_sha256"][path], "current": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()} for path in related},
    "gpu_reexecuted": False,
    "scope": "親讀既有歷史結果與產生程式；沒有重新測量 GPU 或以 CPU 結果取代該結果。",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
