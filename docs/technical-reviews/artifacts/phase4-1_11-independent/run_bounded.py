"""Run only this section's short CPU probe and save real process evidence."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
command = [str(ROOT / ".venv/bin/python"), str(OUT / "bounded_checks.py")]
offline = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
started = time.monotonic()
with (OUT / "bounded-stdout.txt").open("wb") as stdout, (OUT / "bounded-stderr.txt").open("wb") as stderr:
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, **offline}, stdout=stdout, stderr=stderr, timeout=60, check=False)
receipt = {"command": shlex.join(command), "argv": command, "cwd": str(ROOT), "exit_code": result.returncode, "elapsed_seconds": time.monotonic()-started, "timeout_seconds": 60, "offline_environment": offline, "artifacts": {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ("bounded_checks.py", "bounded-stdout.txt", "bounded-stderr.txt")}}
(OUT / "bounded-execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
raise SystemExit(result.returncode)
