"""Record the actual bounded verifier command, environment, stdout and exit status."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
env = os.environ.copy()
overrides = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
             "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
env.update(overrides)
argv = [str(ROOT / ".venv/bin/python"), str(HERE / "verify.py")]
start = datetime.now(timezone.utc).isoformat()
cp = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=30, check=False)
(HERE / "verification.stdout.txt").write_bytes(cp.stdout)
(HERE / "verification.stderr.txt").write_bytes(cp.stderr)
metadata = {"command_argv": argv, "cwd": str(ROOT), "timeout_seconds": 30, "started_utc": start,
            "finished_utc": datetime.now(timezone.utc).isoformat(), "exit_code": cp.returncode,
            "environment_overrides": overrides,
            "code_sha256": hashlib.sha256((HERE / "verify.py").read_bytes()).hexdigest(),
            "stdout_sha256": hashlib.sha256(cp.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(cp.stderr).hexdigest(),
            "initial_failed_attempt": {"code": "verify-failed-attempt.py", "exit_code": 1,
                "stderr": "failed-attempt.stderr.txt", "reason": "Reviewer harness stray unary + before list; corrected before running substantive checks."}}
(HERE / "verification.execution.json").write_text(json.dumps(metadata, indent=2) + "\n")
sys.stdout.buffer.write(cp.stdout)
sys.stderr.buffer.write(cp.stderr)
raise SystemExit(cp.returncode)
