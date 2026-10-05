"""Record real bounded CPU commands, exit status, stdout, stderr and environment."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
overrides = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
             "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
environment = dict(os.environ)
environment.update(overrides)
results = []
for name in ["cpu_checks.py", "verify_ultrafeedback.py"]:
    command = [str(ROOT / ".venv/bin/python"), str(BASE / name)]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, timeout=30)
    stdout = BASE / (name + ".stdout.txt")
    stderr = BASE / (name + ".stderr.txt")
    stdout.write_bytes(completed.stdout)
    stderr.write_bytes(completed.stderr)
    results.append({"command_argv": command, "cwd": str(ROOT), "exit_code": completed.returncode,
                    "elapsed_seconds": time.perf_counter() - started, "timeout_seconds": 30,
                    "environment_overrides": overrides, "python_executable": sys.executable, "python": sys.version,
                    "stdout": str(stdout.relative_to(ROOT)), "stderr": str(stderr.relative_to(ROOT)),
                    "code_sha256": hashlib.sha256((BASE / name).read_bytes()).hexdigest()})
    print(json.dumps(results[-1], ensure_ascii=False))
(BASE / "bounded-execution.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
raise SystemExit(1 if any(result["exit_code"] != 0 for result in results) else 0)
