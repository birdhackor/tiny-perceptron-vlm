"""Historical device-guidance recheck: parser/dispatch only, no neural computation."""
from pathlib import Path
from datetime import datetime, timezone
from contextlib import redirect_stdout
from unittest.mock import patch
import hashlib
import inspect
import io
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = {"created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": "/root/omni_p6_technical_training", "environment": {"python": platform.python_version(), "device": "CPU", "operation": "argparse/default dispatch and simulated availability only; no model forward, training, final or teacher"}, "commands": []}
env = dict(os.environ, PYTHONPATH=str(ROOT))
for script, args in [("quantize.py", ["unused-parser-only.pt", "--device", "cpu"]), ("prepare_data.py", ["--kind", "toy-text", "--device", "cpu"]), ("fetch_training_assets.py", ["--list", "--device", "cpu"])]:
    argv = [sys.executable, "scripts/" + script, *args]
    p = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    out["commands"].append({"argv": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
    assert p.returncode == 2 and "unrecognized arguments: --device cpu" in p.stderr

import torch
from tiny_perceptron.training import choose_device, load_checkpoint
from scripts.course_experiments import run
from scripts import quantize
out["environment"]["torch"] = str(torch.__version__)
out["device_priority_simulation"] = []
for cuda, mps in [(True, True), (False, True), (False, False)]:
    with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
        row = {"simulated_cuda_available": cuda, "simulated_mps_available": mps, "auto": str(choose_device()), "explicit_cpu": str(choose_device("cpu"))}
        assert row["auto"] == ("cuda" if cuda else "mps" if mps else "cpu")
        assert row["explicit_cpu"] == "cpu"
        out["device_priority_simulation"].append(row)
out["simulation_scope"] = "Simulated availability verifies route selection, never actual CUDA/MPS execution."
calls = []
def intercept_execute(experiment_id, device, output, dependencies, assets, **kwargs):
    calls.append({"experiment_id": experiment_id, "device": device, "output": str(output), "dependencies": str(dependencies)})
    return {"experiment_id": experiment_id, "elapsed_seconds": 0.0, "evidence_status": "not_executed_parser_interception", "artifacts": []}
with patch.object(run, "execute", side_effect=intercept_execute), patch.object(sys, "argv", ["run", "--experiment", "sft"]), redirect_stdout(io.StringIO()):
    run.main()
assert calls[0]["device"] == "cpu"
out["experiment_default_parser_dispatch"] = calls
out["quantize_loader_default"] = str(inspect.signature(load_checkpoint).parameters["device"].default)
out["quantize_loader_calls"] = []
class StopBeforeLoad(Exception):
    pass
def intercept_load(*args, **kwargs):
    out["quantize_loader_calls"].append({"args": [str(v) for v in args], "kwargs": kwargs})
    raise StopBeforeLoad
with patch.object(quantize, "load_checkpoint", side_effect=intercept_load), patch.object(sys, "argv", ["quantize", "unused-parser-only.pt"]):
    try:
        quantize.main()
    except StopBeforeLoad:
        pass
assert out["quantize_loader_default"] == "cpu"
assert out["quantize_loader_calls"] == [{"args": ["unused-parser-only.pt"], "kwargs": {}}]
frozen = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v18/sources/training.md"
out["current_source_sha256"] = sha(frozen)
out["current_device_paragraph"] = frozen.read_text().splitlines()[8]
out["source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "scripts/train.py", ROOT / "scripts/evaluate.py", ROOT / "scripts/infer.py", ROOT / "tiny_perceptron/training.py", ROOT / "scripts/quantize.py", ROOT / "scripts/prepare_data.py", ROOT / "scripts/fetch_training_assets.py", ROOT / "scripts/course_experiments/run.py"]}
destination = Path(__file__).with_name("historical-group-9-execution-v1.json")
destination.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"path": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size, "sha256": sha(destination), "result": "all assertions passed; no training/model/teacher/final execution"}, ensure_ascii=False))
