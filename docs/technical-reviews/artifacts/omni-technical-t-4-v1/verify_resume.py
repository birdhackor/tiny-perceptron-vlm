"""Bounded four-update CPU comparison for T.4's interruption/new-stage distinction."""

import hashlib
import json
import os
import platform
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = Path("/workspace/work/omni-technical-t-4-v1/run")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
base = ["uv", "run", "--project", str(ROOT), "--extra", "cpu", "python", "scripts/train.py", "--task", "text", "--data", "data/generated/toy-text/train.jsonl", "--seed", "42", "--device", "cpu", "--train", "--steps", "4"]
report = {"created_at": datetime.now(UTC).isoformat(), "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "threads": "1"}, "commands": []}
for args in (["--checkpoint", "checkpoints/start.pt", "--output", "checkpoints/short-full.pt"], ["--checkpoint", "checkpoints/start.pt", "--stop-after", "2", "--output", "checkpoints/short-resumed.pt"], ["--checkpoint", "checkpoints/short-resumed.pt", "--resume", "--output", "checkpoints/short-resumed.pt"]):
    cmd = base + args
    process = subprocess.run(cmd, cwd=RUN, env=env, text=True, capture_output=True, timeout=30)
    report["commands"].append({"command": shlex.join(cmd), "cwd": str(RUN), "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
    assert process.returncode == 0, report["commands"][-1]
full = torch.load(RUN / "checkpoints/short-full.pt", map_location="cpu", weights_only=True)
resumed = torch.load(RUN / "checkpoints/short-resumed.pt", map_location="cpu", weights_only=True)
difference = max((full["model"][key] - resumed["model"][key]).abs().max().item() for key in full["model"])
assert difference == 0 and full["step"] == resumed["step"] == 4
assert full["python_rng"] == resumed["python_rng"]
assert torch.equal(full["torch_rng"], resumed["torch_rng"])
report["result"] = {"full_steps": full["step"], "interrupted_at": 2, "resumed_steps": resumed["step"], "maximum_weight_difference": difference, "python_rng_identical": True, "torch_rng_identical": True, "full_optimizer_step_values": sorted({int(value["step"].item()) for value in full["optimizer"]["state"].values()}), "resumed_optimizer_step_values": sorted({int(value["step"].item()) for value in resumed["optimizer"]["state"].values()}), "schedule_steps": resumed["metadata"]["schedule_steps"], "full_checkpoint_sha256": hashlib.sha256((RUN / "checkpoints/short-full.pt").read_bytes()).hexdigest(), "resumed_checkpoint_sha256": hashlib.sha256((RUN / "checkpoints/short-resumed.pt").read_bytes()).hexdigest()}
report["scope"] = "Only four CPU updates with the lesson baseline and data; byte-level parameter/RNG equality in this installed environment. No broad hardware/version reproducibility claim and no fixed recipe execution."
report["completed_at"] = datetime.now(UTC).isoformat()
(OUT / "resume-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report["result"], ensure_ascii=False))
