"""Save real command, stdout, stderr, exit status and hashes for this audit."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
commands = {
    "standalone-original": [str(ROOT / ".venv/bin/python"), "-I", str(BASE / "original/fence-1.py")],
    "bounded-cpu": [str(ROOT / ".venv/bin/python"), str(OUT / "bounded_cpu.py")],
}
records = []
for name, argv in commands.items():
    started = time.perf_counter()
    completed = subprocess.run(argv, cwd=ROOT, env=environment, capture_output=True, timeout=45, check=False)
    (OUT / f"{name}.stdout.txt").write_bytes(completed.stdout)
    (OUT / f"{name}.stderr.txt").write_bytes(completed.stderr)
    record = {"id": name, "command_argv": argv, "command": shlex.join(argv), "cwd": str(ROOT), "timeout_seconds": 45, "exit_code": completed.returncode, "elapsed_seconds": time.perf_counter()-started, "environment_overrides": {k: environment[k] for k in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE"]}, "stdout_path": f"runs/{name}.stdout.txt", "stderr_path": f"runs/{name}.stderr.txt", "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(), "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(), "script_sha256": hashlib.sha256(Path(argv[-1]).read_bytes()).hexdigest()}
    records.append(record)
    (OUT / "execution-receipts.json").write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(record, ensure_ascii=False))
    print(completed.stdout.decode())
    if completed.returncode:
        print(completed.stderr.decode())
        raise SystemExit(completed.returncode)
