"""Actual bounded CPU invocation; never resolve the venv executable symlink."""
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
argv = [str(ROOT / ".venv/bin/python"), str(BASE / "verify_current.py")]
env = dict(os.environ)
offline = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
env.update(offline)
start = time.perf_counter()
with (BASE / "stdout.txt").open("wb") as out, (BASE / "stderr.txt").open("wb") as err:
    result = subprocess.run(argv, cwd=ROOT, env=env, stdout=out, stderr=err, timeout=30, check=False)
metadata = {"command_argv": argv, "command": " ".join(argv), "cwd": str(ROOT), "shell": "bash", "login": False, "timeout_seconds": 30, "exit_code": result.returncode, "elapsed_seconds": time.perf_counter() - start, "offline_environment": offline}
(BASE / "execution.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
print((BASE / "stdout.txt").read_text())
print((BASE / "stderr.txt").read_text())
raise SystemExit(result.returncode)
