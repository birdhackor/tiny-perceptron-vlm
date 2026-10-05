"""Persist actual commands, stdout, stderr, exit status, and environment."""
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT))
environment = dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__), torch_git_version=str(torch.version.git_version), cuda_build=str(torch.version.cuda), cuda_available=str(torch.cuda.is_available()), device="cpu", platform=platform.platform(), offline_overrides={k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE"]})
(HERE / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
results = []
for label, name in [("original", "fence-1.py"), ("variants", "cpu_checks.py")]:
    command = [sys.executable, str(HERE / name)]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=60, check=False)
    (HERE / (label + ".stdout.txt")).write_bytes(result.stdout)
    (HERE / (label + ".stderr.txt")).write_bytes(result.stderr)
    results.append(dict(label=label, command=shlex.join(command), cwd=str(ROOT), exit_code=result.returncode, elapsed_seconds=time.perf_counter() - started, timeout_seconds=60, code_sha256=hashlib.sha256((HERE / name).read_bytes()).hexdigest(), stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stderr_sha256=hashlib.sha256(result.stderr).hexdigest()))
    print(label, "exit", result.returncode)
    print(result.stdout.decode(), end="")
    if result.stderr:
        print(result.stderr.decode(), end="")
(HERE / "execution.json").write_text(json.dumps(results, indent=2) + "\n")
sys.exit(0 if all(item["exit_code"] == 0 for item in results) else 1)
