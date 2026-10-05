"""Capture exact bounded CPU commands, exit status and permanent output hashes."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[3]
env_changes = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
commands = [("original-direct", BASE / "inputs/fence-1.py"), ("bounded-probe", BASE / "probe.py")]
receipts = []
for label, script in commands:
    argv = [str(REPO / ".venv/bin/python"), str(script)]
    stdout = BASE / "execution" / f"{label}.stdout.txt"
    stderr = BASE / "execution" / f"{label}.stderr.txt"
    start = time.perf_counter()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        completed = subprocess.run(argv, cwd=REPO, env={**os.environ, **env_changes},
                                   stdout=out, stderr=err, check=False, timeout=45)
    receipts.append({"label": label, "command_argv": argv, "command": shlex.join(argv), "cwd": str(REPO),
                     "environment_overrides": env_changes, "timeout_seconds": 45,
                     "exit_code": completed.returncode, "elapsed_seconds": time.perf_counter() - start,
                     "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                     "stdout": str(stdout.relative_to(REPO)), "stdout_sha256": hashlib.sha256(stdout.read_bytes()).hexdigest(),
                     "stderr": str(stderr.relative_to(REPO)), "stderr_sha256": hashlib.sha256(stderr.read_bytes()).hexdigest()})
    if completed.returncode != 0:
        print(stderr.read_text())
        raise SystemExit(completed.returncode)
(BASE / "execution/command-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
