"""Run the exact independent exercise and bounded contracts with receipts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[3]
PYTHON = REPO / ".venv/bin/python"
commands = [
    ("independent-exercise", [str(PYTHON), "-I", str(BASE / "original-run/fence-2.py")]),
    ("contracts", [str(PYTHON), "-I", str(BASE / "code/check_contracts.py")]),
]
receipts = []
for name, argv in commands:
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(argv, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
    (BASE / f"{name}.stdout.txt").write_bytes(completed.stdout)
    (BASE / f"{name}.stderr.txt").write_bytes(completed.stderr)
    receipt = {"name": name, "command_argv": argv, "cwd": str(REPO), "started_utc": started,
        "exit_code": completed.returncode, "timeout_seconds": 60,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "python": sys.version, "input_code_sha256": hashlib.sha256(Path(argv[-1]).read_bytes()).hexdigest()}
    receipts.append(receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    if completed.returncode:
        print(completed.stderr.decode(), file=sys.stderr)
        raise SystemExit(completed.returncode)
(BASE / "bounded-run-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
