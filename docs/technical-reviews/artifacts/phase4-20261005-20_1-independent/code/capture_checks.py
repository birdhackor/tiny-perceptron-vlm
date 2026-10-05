"""Run each bounded reviewer check and retain exact commands and process output."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
records = []
for name, timeout in [("verify_routes", 30), ("verify_selection", 15), ("render_figure", 30)]:
    command = [str(ROOT / ".venv/bin/python"), str(BASE / "code" / (name + ".py"))]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=timeout)
    (BASE / "execution" / (name + ".stdout.txt")).write_bytes(completed.stdout)
    (BASE / "execution" / (name + ".stderr.txt")).write_bytes(completed.stderr)
    record = {"name": name, "command_argv": command, "command": shlex.join(command), "cwd": str(ROOT), "exit_code": completed.returncode, "elapsed_seconds": time.perf_counter() - started, "timeout_seconds": timeout, "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(), "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest()}
    records.append(record)
    print(json.dumps(record))
    if completed.returncode:
        print(completed.stderr.decode())
        raise SystemExit(completed.returncode)
environment = {"python": sys.version, "executable": sys.executable, "device": "cpu", "torch": importlib.metadata.version("torch"), "numpy": importlib.metadata.version("numpy"), "Pillow": importlib.metadata.version("pillow"), "soundfile": importlib.metadata.version("soundfile"), "playwright": importlib.metadata.version("playwright"), "offline_flags": {k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]}}
(BASE / "execution/captured-checks.json").write_text(json.dumps({"runs": records, "environment": environment}, ensure_ascii=False, indent=2) + "\n")
