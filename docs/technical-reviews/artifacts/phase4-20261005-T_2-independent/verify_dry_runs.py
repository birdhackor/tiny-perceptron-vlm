import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys

import torch
import tiny_perceptron.model as model_helpers
import tiny_perceptron.training as training_helpers

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
LOGS = OUT / "execution"
LOGS.mkdir(exist_ok=True)
assert torch.version.cuda is None
ENV = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "cuda_build": str(torch.version.cuda),
    "device": "cpu",
    "cpu_threads_per_command": "main sets min(4, current_threads)",
}
offline = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
os.environ.update(offline)
original_sha = hashlib.sha256((ROOT / "scripts/train.py").read_bytes()).hexdigest()
records = []
for task in ("text", "sft", "vision"):
    argv = [str(ROOT / ".venv/bin/python"), "scripts/train.py", "--task", task, "--device", "cpu"]
    result = subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), text=True, capture_output=True, timeout=45)
    (LOGS / f"{task}.stdout.jsonl").write_text(result.stdout)
    (LOGS / f"{task}.stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 2
    entry, report = lines
    assert report["mode"] == "dry-run-no-weight-update"
    assert report["task"] == task and report["device"] == "cpu"
    assert len(report["history"]) == 1 and report["history"][0] == entry
    assert math.isfinite(entry["loss"])
    assert math.isfinite(entry["grad_norm"]) and entry["grad_norm"] > 0
    assert entry["effective_tokens"] is None if task == "vision" else entry["effective_tokens"] > 0

    # The second bounded run executes the same unmodified entry point. Hooks only
    # observe loss labels, gradients and parameter bytes; unexpected updates/saves fail.
    observations = {"optimizer_step_calls": 0, "save_checkpoint_calls": 0, "torch_save_calls": 0, "loss_labels": []}
    parameter_snapshots = []
    opt_init, opt_step = torch.optim.AdamW.__init__, torch.optim.AdamW.step
    checkpoint_save, tensor_save = training_helpers.save_checkpoint, torch.save
    masked_loss, clip_norm = model_helpers.masked_loss, torch.nn.utils.clip_grad_norm_

    def observe_init(self, params, *args, **kwargs):
        params = list(params)
        parameter_snapshots.extend((p, p.detach().clone()) for p in params)
        return opt_init(self, params, *args, **kwargs)

    def prohibit_step(*args, **kwargs):
        observations["optimizer_step_calls"] += 1
        raise AssertionError("Dry-run called optimizer.step")

    def prohibit_checkpoint(*args, **kwargs):
        observations["save_checkpoint_calls"] += 1
        raise AssertionError("Dry-run saved checkpoint")

    def prohibit_save(*args, **kwargs):
        observations["torch_save_calls"] += 1
        raise AssertionError("Dry-run called torch.save")

    def observe_loss(logits, labels):
        observations["loss_labels"].append({"logits_shape": list(logits.shape), "labels_shape": list(labels.shape), "scored_positions": int((labels != -100).sum())})
        return masked_loss(logits, labels)

    def observe_norm(parameters, *args, **kwargs):
        parameters = list(parameters)
        grads = [p.grad for p in parameters if p.grad is not None]
        manual_l2 = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
        value = clip_norm(parameters, *args, **kwargs)
        observations["gradient_norm"] = {
            "parameter_tensors": len(parameters),
            "with_gradient": len(grads),
            "without_gradient": len(parameters) - len(grads),
            "with_nonzero_gradient": sum(bool(g.ne(0).any()) for g in grads),
            "manual_float64_l2_before_clipping": manual_l2,
            "reported_float32_l2_before_clipping": float(value),
        }
        assert math.isclose(manual_l2, float(value), rel_tol=2e-6, abs_tol=2e-6)
        return value

    torch.optim.AdamW.__init__, torch.optim.AdamW.step = observe_init, prohibit_step
    training_helpers.save_checkpoint, torch.save = prohibit_checkpoint, prohibit_save
    model_helpers.masked_loss, torch.nn.utils.clip_grad_norm_ = observe_loss, observe_norm
    old_argv = sys.argv
    stream = io.StringIO()
    try:
        sys.argv = ["scripts/train.py", "--task", task, "--device", "cpu"]
        with contextlib.redirect_stdout(stream):
            runpy.run_path(str(ROOT / "scripts/train.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
        torch.optim.AdamW.__init__, torch.optim.AdamW.step = opt_init, opt_step
        training_helpers.save_checkpoint, torch.save = checkpoint_save, tensor_save
        model_helpers.masked_loss, torch.nn.utils.clip_grad_norm_ = masked_loss, clip_norm
    observed_lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    assert observed_lines[0] == entry
    assert parameter_snapshots
    changed = sum(not torch.equal(p.detach(), initial) for p, initial in parameter_snapshots)
    assert changed == 0
    observations["changed_parameter_tensors"] = changed
    observations["captured_parameter_tensors"] = len(parameter_snapshots)
    denominator = sum(item["scored_positions"] for item in observations["loss_labels"])
    observations["scored_positions_in_batch"] = denominator
    assert denominator > 0
    if task != "vision":
        assert denominator == entry["effective_tokens"]
    (LOGS / f"{task}.observed-original.stdout.jsonl").write_text(stream.getvalue())
    records.append({"task": task, "command_argv": argv, "exit_code": result.returncode, "raw_stdout": f"execution/{task}.stdout.jsonl", "report_pointer": "/history/0", "entry": entry, "observations": observations})

# A tiny independent change distinguishes computing gradients from updates and
# demonstrates that a positive total norm can coexist with an unused parameter.
p = torch.nn.Parameter(torch.tensor(2.0))
unused = torch.nn.Parameter(torch.tensor(7.0))
before = p.detach().clone()
loss = (p - 1.0).square()
loss.backward()
assert torch.equal(p.detach(), before) and p.grad.item() == 2.0 and unused.grad is None
norm = float(torch.nn.utils.clip_grad_norm_([p, unused], 1.0))
assert norm == 2.0 and torch.equal(p.detach(), before)
small_change = {"loss": float(loss.detach()), "parameter_before": float(before), "parameter_after_backward_and_clip": float(p.detach()), "gradient_before_clipping": 2.0, "returned_total_norm": norm, "unused_parameter_gradient": None, "supports": "Backward and clipping mutate gradients, not parameter values; a positive norm does not imply every parameter has a gradient."}
assert hashlib.sha256((ROOT / "scripts/train.py").read_bytes()).hexdigest() == original_sha
summary = {"environment": ENV, "original_train_sha256": original_sha, "offline_environment": offline, "records": records, "bounded_variation": small_change, "scope": "Only default one-batch dry-runs and a two-scalar gradient check; no --train, model/data downloads, saved weights, full training, or capability evaluation."}
(LOGS / "verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
