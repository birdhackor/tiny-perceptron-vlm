"""Persist a real, bounded CPU execution receipt and output hashes."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
script = HERE / "verify_chat_contract.py"
command = [str(ROOT / ".venv/bin/python"), str(script)]
settings = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
started = time.perf_counter()
with (HERE / "verification.stdout.txt").open("wb") as stdout, (HERE / "verification.stderr.txt").open("wb") as stderr:
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, **settings}, stdout=stdout, stderr=stderr, timeout=30, check=False)
receipt = {"command_argv": command, "command": shlex.join(command), "cwd": str(ROOT), "timeout_seconds": 30, "elapsed_seconds": time.perf_counter() - started, "exit_code": result.returncode, "environment_overrides": settings, "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "input_provenance": "current repository helpers personally inspected and snapshotted under inputs; deterministic source text, no dataset/checkpoint/model inputs", "artifacts": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ("verification.stdout.txt", "verification.stderr.txt")}}
(HERE / "verification.execution.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
sys.exit(result.returncode)
