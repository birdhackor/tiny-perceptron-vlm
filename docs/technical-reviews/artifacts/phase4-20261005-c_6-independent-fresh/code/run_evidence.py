"""Capture real stdout, stderr, command, environment and child exit status."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
mode = sys.argv[1]
script = {"bounded": "verify_c6.py", "source-identity": "source_identity_check.py"}[mode]
command = [sys.executable, str(BASE / "code" / script)]
env = dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
if mode == "bounded":
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
with (BASE / "execution" / f"{mode}.stdout.txt").open("wb") as stdout, (BASE / "execution" / f"{mode}.stderr.txt").open("wb") as stderr:
    completed = subprocess.run(command, cwd=Path.cwd(), env=env, stdout=stdout, stderr=stderr, timeout=60, check=False)
result = {"command": shlex.join(command), "command_argv": command, "cwd": str(Path.cwd()),
          "exit_code": completed.returncode, "python": sys.version, "device_requested": "cpu",
          "script_sha256": hashlib.sha256((BASE / "code" / script).read_bytes()).hexdigest(),
          "stdout": f"{mode}.stdout.txt", "stderr": f"{mode}.stderr.txt"}
(BASE / "execution" / f"{mode}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
sys.exit(completed.returncode)
