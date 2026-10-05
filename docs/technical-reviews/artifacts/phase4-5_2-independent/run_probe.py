"""Capture the actual bounded CPU command, exit status, outputs, and source hashes."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
command = [str(ROOT / ".venv/bin/python"), str(BASE / "cpu_probe.py")]
environment = dict(os.environ)
offline = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
environment.update(offline)
began = time.perf_counter()
with (BASE / "cpu-stdout.txt").open("wb") as stdout, (BASE / "cpu-stderr.txt").open("wb") as stderr:
    completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, timeout=30, check=False)
receipt = {"command_argv": command, "command": "CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/phase4-5_2-independent/cpu_probe.py", "cwd": str(ROOT), "elapsed_seconds": time.perf_counter() - began, "timeout_seconds": 30, "exit_code": completed.returncode, "offline_variables": offline, "inputs": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ("tiny_perceptron/model.py", "tiny_perceptron/data.py")}, "local_code_sha256": hashlib.sha256((BASE / "cpu_probe.py").read_bytes()).hexdigest(), "output_sha256": {p: hashlib.sha256((BASE / p).read_bytes()).hexdigest() for p in ("cpu-stdout.txt", "cpu-stderr.txt", "cpu-results.json")}}
(BASE / "cpu-execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
raise SystemExit(completed.returncode)
