import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
command = [str(ROOT / ".venv/bin/python"), str(HERE / "inspect-context.py")]
env = dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
started = time.perf_counter()
with (HERE / "inspection-stdout.txt").open("wb") as stdout, (HERE / "inspection-stderr.txt").open("wb") as stderr:
    p = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=20, check=False)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {"command_argv": command, "command": shlex.join(command), "cwd": str(ROOT), "exit_code": p.returncode,
    "elapsed_seconds": time.perf_counter() - started,
    "environment": json.loads((HERE / "environment.json").read_bytes()) if (HERE / "environment.json").exists() else {},
    "scope": "Actual same-owner current 11.2 and necessary 5.1 hash/AST inspection; prior neural/scalar CPU proof was not rerun.",
    "files": {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "inspection-receipt.json"}}
(HERE / "inspection-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"exit_code": p.returncode, "receipt_sha256": sha(HERE / "inspection-receipt.json"), "files": receipt["files"]}, indent=2))
raise SystemExit(p.returncode)
