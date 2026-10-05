"""Record real commands, stdout/stderr, exit status, and versions for bounded evidence."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
jobs = [
    ("cpu", [str(ROOT / ".venv/bin/python"), str(HERE / "verify_cpu.py")], 35),
    ("prerequisite-figure", ["inkscape", str(HERE.parent / "figures/prerequisite-8_8-lora-paths.svg"),
          "--export-type=png", "--export-filename=" + str(HERE.parent / "figures/prerequisite-8_8-lora-paths.png")], 20),
    ("paper-page4", ["pdftoppm", "-f", "4", "-l", "4", "-singlefile", "-scale-to", "1600", "-png",
        str(HERE.parent / "sources/lora-2106.09685v2.pdf"), str(HERE.parent / "sources/lora-page4")], 20),
]
receipts = []
for name, argv, timeout in jobs:
    started = time.perf_counter()
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, timeout=timeout)
    (HERE / (name + "-stdout.txt")).write_bytes(result.stdout)
    (HERE / (name + "-stderr.txt")).write_bytes(result.stderr)
    receipt = {"name": name, "command_argv": argv, "cwd": str(ROOT), "exit_code": result.returncode,
       "elapsed_seconds": time.perf_counter() - started, "timeout_seconds": timeout,
       "python": sys.version, "device": "cpu", "CUDA_VISIBLE_DEVICES": "",
       "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(), "stderr_sha256": hashlib.sha256(result.stderr).hexdigest()}
    receipts.append(receipt)
    print(json.dumps(receipt))
    (HERE / "bounded-command-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    if result.returncode:
        print(result.stderr.decode(), file=sys.stderr)
        raise SystemExit(result.returncode)
