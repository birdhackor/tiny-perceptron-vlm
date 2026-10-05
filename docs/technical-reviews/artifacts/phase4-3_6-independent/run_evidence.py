"""Run bounded checks, recording exact commands and exit status."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def run(name, command, timeout=40, environment=None):
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, timeout=timeout)
        record = {"name": name, "command_argv": command, "cwd": str(ROOT), "exit_code": result.returncode, "timeout_seconds": timeout}
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        record = {"name": name, "command_argv": command, "cwd": str(ROOT), "exit_code": 124, "timed_out": True, "timeout_seconds": timeout}
        stdout, stderr = error.stdout or b"", error.stderr or b""
    (OUT / (name + ".stdout.txt")).write_bytes(stdout)
    (OUT / (name + ".stderr.txt")).write_bytes(stderr)
    record["elapsed_seconds"] = time.perf_counter() - started
    return record

records = []
records.append(run("prepare", [sys.executable, str(OUT / "prepare_evidence.py")]))
assert records[-1]["exit_code"] == 0
cpu_environment = os.environ.copy()
cpu_environment.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
records.append(run("verify", [sys.executable, str(OUT / "verify.py")], environment=cpu_environment))
assert records[-1]["exit_code"] == 0
figure = ROOT / "course/figures/rewrite-03-causal-results.svg"
browser_png = OUT / "causal-chromium.png"
records.append(run("chromium", ["chromium", "--headless", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--allow-file-access-from-files", "--hide-scrollbars", "--window-size=640,1028", "--screenshot=" + str(browser_png), figure.as_uri()], timeout=15))
# Inkscape provides an exact SVG canvas rendering even if Chromium is unavailable.
records.append(run("inkscape-version", ["inkscape", "--version"]))
records.append(run("inkscape", ["inkscape", str(figure), "--export-type=png", "--export-filename=" + str(OUT / "causal-inkscape.png")]))
assert records[-1]["exit_code"] == 0
(OUT / "run-receipts.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(records, ensure_ascii=False, indent=2))
