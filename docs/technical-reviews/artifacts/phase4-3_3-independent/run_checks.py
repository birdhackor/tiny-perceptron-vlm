"""Preserve already executed original-code facts and run bounded changes with receipts."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
ORIGINAL = Path("/tmp/phase4-3_3-original")
(OUT / "original").mkdir(exist_ok=True)
for name in ("section.md", "fence-1.py", "bootstrap.py", "extraction.json", "environment.json", "execution.json", "stdout.txt", "stderr.txt"):
    shutil.copy2(ORIGINAL / name, OUT / "original" / name)
shutil.copy2(ROOT / "docs/review-tools/section_facts.py", OUT / "original" / "section_facts.py")
shutil.copy2(ROOT / "tiny_perceptron/attention.py", OUT / "original" / "attention.py")
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
command = [str(ROOT / ".venv/bin/python"), str(OUT / "probe.py")]
started = time.perf_counter()
with (OUT / "probe-stdout.txt").open("wb") as stdout, (OUT / "probe-stderr.txt").open("wb") as stderr:
    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=30, check=False)
receipt = {"command": shlex.join(command), "cwd": str(ROOT), "exit_code": completed.returncode,
           "elapsed_seconds": time.perf_counter() - started, "timeout_seconds": 30,
           "environment": json.loads((OUT / "probe-results.json").read_text())["environment"] if completed.returncode == 0 else {},
           "inputs": {}, "outputs": {}}
for group, names in (("inputs", ("probe.py", "original/fence-1.py", "original/attention.py")),
                     ("outputs", ("probe-stdout.txt", "probe-stderr.txt", "probe-results.json"))):
    for name in names:
        path = OUT / name
        if path.exists():
            receipt[group][name] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / "probe-execution.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
raise SystemExit(completed.returncode)
