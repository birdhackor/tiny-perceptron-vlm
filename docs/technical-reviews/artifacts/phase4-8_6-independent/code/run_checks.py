"""Capture the actual bounded-check command, environment, stdout and exit code."""
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

repo = Path(__file__).resolve().parents[5]
artifact = Path(__file__).resolve().parents[1]
command = [str(repo / ".venv/bin/python"), str(artifact / "code/check.py")]
offline = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
env = {**os.environ, **offline}
started = time.perf_counter()
with (artifact / "execution/bounded-stdout.txt").open("wb") as out, (artifact / "execution/bounded-stderr.txt").open("wb") as err:
    result = subprocess.run(command, cwd=repo, env=env, stdout=out, stderr=err, timeout=45, check=False)
record = {"command_argv": command, "command": shlex.join(command), "cwd": str(repo),
          "exit_code": result.returncode, "elapsed_seconds": time.perf_counter() - started,
          "environment": {"python": sys.version, "device": "cpu", "platform": platform.platform()},
          "offline_environment": offline, "files": {}}
for name in ["code/check.py", "execution/bounded-results.json", "execution/bounded-stdout.txt", "execution/bounded-stderr.txt"]:
    p = artifact / name
    if p.exists(): record["files"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
(artifact / "execution/bounded-execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(record, ensure_ascii=False, indent=2))
if result.returncode:
    print((artifact / "execution/bounded-stderr.txt").read_text())
raise SystemExit(result.returncode)
