"""Capture the real bounded callback command and exit status, without model execution."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[5]
ART = Path(__file__).resolve().parents[1]
env = dict(os.environ)
restrictions = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
env.update(restrictions)
argv = [str(ROOT / ".venv/bin/python"), str(ART / "code/verify_callback.py")]
start = time.monotonic()
result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=60, check=False)
(ART / "verification.stdout.txt").write_bytes(result.stdout)
(ART / "verification.stderr.txt").write_bytes(result.stderr)
verification = json.loads((ART / "callback-verification.json").read_bytes())
receipt = {"command_argv": argv, "command": shlex.join(argv), "cwd": str(ROOT),
           "exit_code": result.returncode, "elapsed_seconds": time.monotonic()-start,
           "timeout_seconds": 60, "offline_environment": restrictions,
           "environment": verification["environment"],
           "code_sha256": hashlib.sha256((ART / "code/verify_callback.py").read_bytes()).hexdigest(),
           "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
           "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
           "scope": "Hash/version/context/renderer checks only; zero new model forward, gradient or optimizer runs."}
(ART / "callback-execution-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
raise SystemExit(result.returncode)
