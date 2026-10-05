from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
cmd = [str(ROOT / ".venv/bin/python"), str(OUT / "probe.py")]
env = dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT))
start = time.monotonic()
completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, timeout=45)
(OUT / "probe.stdout.txt").write_bytes(completed.stdout)
(OUT / "probe.stderr.txt").write_bytes(completed.stderr)
receipt = {"command": shlex.join(cmd), "cwd": str(ROOT), "exit_code": completed.returncode, "elapsed_seconds": time.monotonic() - start,
 "timeout_seconds": 45, "code_sha256": hashlib.sha256((OUT / "probe.py").read_bytes()).hexdigest(),
 "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(), "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
 "offline_environment": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MKL_NUM_THREADS","HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","TRANSFORMERS_OFFLINE","PYTHONDONTWRITEBYTECODE")}}
(OUT / "probe-execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
print(completed.stdout.decode())
print(completed.stderr.decode())
raise SystemExit(completed.returncode)
