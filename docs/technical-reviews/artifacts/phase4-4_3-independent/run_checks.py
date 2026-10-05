"""Capture actual exit status and hashes for the bounded probe and source PDF inspection."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
commands = [
    ("probe", [sys.executable, str(ROOT / "verify_layernorm.py")]),
    ("pdf-extract", ["pdftotext", "-layout", str(ROOT / "ba-2016-layer-normalization-v1.pdf"), str(ROOT / "ba-2016-layer-normalization-v1.txt")]),
    ("pdf-render", ["pdftoppm", "-f", "2", "-l", "3", "-scale-to", "1600", "-png", str(ROOT / "ba-2016-layer-normalization-v1.pdf"), str(ROOT / "paper-page")]),
]
env = dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
receipts = []
for name, command in commands:
    start = time.perf_counter()
    with (ROOT / f"{name}-stdout.txt").open("wb") as stdout, (ROOT / f"{name}-stderr.txt").open("wb") as stderr:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, timeout=60, check=False)
    receipts.append({"name": name, "command": shlex.join(command), "cwd": str(REPO), "exit_code": result.returncode,
                     "elapsed_seconds": time.perf_counter()-start,
                     "stdout_sha256": hashlib.sha256((ROOT / f"{name}-stdout.txt").read_bytes()).hexdigest(),
                     "stderr_sha256": hashlib.sha256((ROOT / f"{name}-stderr.txt").read_bytes()).hexdigest()})
(ROOT / "checks-execution.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
assert all(receipt["exit_code"] == 0 for receipt in receipts)
