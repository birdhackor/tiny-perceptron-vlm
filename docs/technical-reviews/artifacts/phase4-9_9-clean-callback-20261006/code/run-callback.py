"""Persist this callback's actual bounded inspection command/stdout/env/hash receipt."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
A = Path(__file__).resolve().parents[1]
code = A / "code/callback-inspect.py"
cmd = [str(ROOT / ".venv/bin/python"), str(code)]
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
started = time.perf_counter()
r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, timeout=30)
(A / "execution/callback-inspect.stdout.txt").write_bytes(r.stdout)
(A / "execution/callback-inspect.stderr.txt").write_bytes(r.stderr)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = {"command_argv":cmd, "cwd":str(ROOT), "timeout_seconds":30, "exit_code":r.returncode,
           "elapsed_seconds":time.perf_counter()-started, "code_sha256":sha(code),
           "stdout_path":str((A / "execution/callback-inspect.stdout.txt").relative_to(ROOT)), "stdout_sha256":sha(A / "execution/callback-inspect.stdout.txt"),
           "stderr_path":str((A / "execution/callback-inspect.stderr.txt").relative_to(ROOT)), "stderr_sha256":sha(A / "execution/callback-inspect.stderr.txt"),
           "environment":{"python":sys.version, "python_executable":sys.executable, "device":"CPU-only file/Unicode/hash inspection; no torch/model execution"},
           "offline_overrides":{k:env[k] for k in ["CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"]}}
if r.returncode == 0:
    receipt["actual_inspection_path"] = str((A / "execution/actual-inspection.json").relative_to(ROOT))
    receipt["actual_inspection_sha256"] = sha(A / "execution/actual-inspection.json")
(A / "execution/callback-execution-receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
print(r.stdout.decode())
print(r.stderr.decode())
raise SystemExit(r.returncode)
