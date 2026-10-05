"""Run unchanged training CLIs and observe parameter updates in bounded CPU runs."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import runpy
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parent
WORK = ROOT / "outputs/natural-v4/factual-research/W.7"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
WORK.mkdir(parents=True, exist_ok=True)
PYTHON = ROOT / ".venv/bin/python"
records = []
for name, argv in [
    ("text-default-dry", ["scripts/train.py", "--task", "text", "--device", "cpu"]),
    ("sft-default-dry", ["scripts/train.py", "--task", "sft", "--device", "cpu"]),
    ("bigram-default-dry", ["scripts/train_simple.py", "--model", "bigram", "--seed", "42", "--device", "cpu"]),
    ("text-one-step-cli", ["scripts/train.py", "--task", "text", "--device", "cpu", "--train", "--steps", "1", "--batch-size", "1", "--width", "8", "--max-length", "32", "--output", str(WORK / "cli-one-step.pt")]),
]:
    command = [str(PYTHON), "-B", *argv]
    completed = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=40)
    stdout_path = ART / (name + ".stdout.txt")
    stderr_path = ART / (name + ".stderr.txt")
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    assert completed.returncode == 0, (name, completed.stderr)
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    expected_mode = "train" if "--train" in argv else "dry-run-no-weight-update"
    assert report["mode"] == expected_mode
    if "history" in report:
        assert len(report["history"]) == 1
        assert math.isfinite(report["history"][0]["loss"])
        assert math.isfinite(report["history"][0]["grad_norm"]) and report["history"][0]["grad_norm"] > 0
    records.append({"name": name, "argv": command, "cwd": str(ROOT), "exit_code": completed.returncode, "stdout_path": stdout_path.relative_to(ROOT).as_posix(), "stderr_path": stderr_path.relative_to(ROOT).as_posix(), "report": report})

# Observe actual AdamW calls and weights through runpy; the script bytes are unchanged.
original_adamw = torch.optim.AdamW
optimizers = []
class RecordingAdamW(original_adamw):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.review_steps = 0
        self.review_parameters = [p for group in self.param_groups for p in group["params"]]
        self.review_before = [p.detach().clone() for p in self.review_parameters]
        optimizers.append(self)
    def step(self, *args, **kwargs):
        self.review_steps += 1
        return super().step(*args, **kwargs)

sys.path.insert(0, str(ROOT))
old_argv = sys.argv
torch.optim.AdamW = RecordingAdamW
observations = []
try:
    for name, train in [("observed-text-dry", False), ("observed-text-one-step", True)]:
        target = WORK / (name + ".pt")
        assert not target.exists(), "Refusing to reuse an existing checkpoint"
        sys.argv = [str(ROOT / "scripts/train.py"), "--task", "text", "--device", "cpu", "--steps", "1", "--batch-size", "1", "--width", "8", "--max-length", "32", "--seed", "42", "--output", str(target)]
        if train:
            sys.argv.append("--train")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")
        (ART / (name + ".stdout.txt")).write_text(stdout.getvalue())
        (ART / (name + ".stderr.txt")).write_text(stderr.getvalue())
        optimizer = optimizers[-1]
        changed = sum(not torch.equal(before, p.detach()) for before, p in zip(optimizer.review_before, optimizer.review_parameters))
        max_change = max(float((before - p.detach()).abs().max()) for before, p in zip(optimizer.review_before, optimizer.review_parameters))
        report = json.loads(stdout.getvalue().strip().splitlines()[-1])
        observed = {"name": name, "argv": list(sys.argv), "instrumentation": "Read-only runpy execution with AdamW subclass that records before tensors and calls original step", "optimizer_step_calls": optimizer.review_steps, "parameter_tensors": len(optimizer.review_parameters), "scalar_parameters": sum(p.numel() for p in optimizer.review_parameters), "changed_parameter_tensors": changed, "max_abs_parameter_change": max_change, "checkpoint_exists": target.exists(), "sidecar_exists": target.with_suffix(".json").exists(), "report": report}
        assert optimizer.review_steps == int(train)
        assert target.exists() == train and target.with_suffix(".json").exists() == train
        assert (changed > 0 and max_change > 0) if train else (changed == 0 and max_change == 0)
        if train:
            payload = torch.load(target, weights_only=True, map_location="cpu")
            assert payload["step"] == 1 and payload["metadata"]["report"]["mode"] == "train"
            observed["checkpoint_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            observed["saved_step"] = payload["step"]
        observations.append(observed)
finally:
    torch.optim.AdamW = original_adamw
    sys.argv = old_argv

result = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"}, "commands": records, "parameter_observations": observations, "scope": "Four unchanged CLI runs plus two instrumented one-batch/one-step CPU checks. These verify CLI gating and updates, not training quality or full experiments.", "denominators": {"unchanged_cli_runs": len(records), "instrumented_runs": len(observations), "dry_updates": 0, "trained_updates_per_run": 1, "instrumented_batch_size": 1, "instrumented_seed": 42, "instrumented_effective_tokens": [r["report"]["history"][0]["effective_tokens"] for r in observations]}}
(ART / "training-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"environment": result["environment"], "denominators": result["denominators"], "observations": [{k: r[k] for k in ["name", "optimizer_step_calls", "scalar_parameters", "changed_parameter_tensors", "max_abs_parameter_change", "checkpoint_exists"]} for r in observations], "result": "all bounded CLI and update assertions passed"}, indent=2))
