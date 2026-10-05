"""Record actual bounded CPU commands and exit status for independent 19.8 review."""
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py", "bootstrap.py", "section.md"):
    destination = HERE / "original-fence" / name
    destination.parent.mkdir(exist_ok=True)
    shutil.copyfile(Path("/tmp/phase4-19_8-factual-fence-run") / name, destination)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
commands = [
    ("verify", [str(ROOT / ".venv/bin/python"), str(HERE / "verify_cpu.py")]),
    ("reference-test", [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/test_capstone.py::test_dpo_reference_frozen_and_same_facts_not_style_fabrication"]),
]
receipts = []
for name, argv in commands:
    started = time.monotonic()
    with (HERE / f"{name}.stdout.txt").open("wb") as stdout, (HERE / f"{name}.stderr.txt").open("wb") as stderr:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=60, check=False)
    receipt = {"name": name, "command": shlex.join(argv), "argv": argv, "cwd": str(ROOT),
               "exit_code": result.returncode, "elapsed_seconds": time.monotonic()-started,
               "timeout_seconds": 60, "environment": {
                   "python": platform.python_version(), "device": "cpu", "CUDA_VISIBLE_DEVICES": "empty",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "network": "offline variables; no downloading commands"},
               "stdout_sha256": hashlib.sha256((HERE / f"{name}.stdout.txt").read_bytes()).hexdigest(),
               "stderr_sha256": hashlib.sha256((HERE / f"{name}.stderr.txt").read_bytes()).hexdigest()}
    receipts.append(receipt)
    print(json.dumps(receipt))
(HERE / "command-results.json").write_text(json.dumps(receipts, indent=2) + "\n")
if any(r["exit_code"] for r in receipts):
    raise SystemExit(1)
