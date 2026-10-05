"""Archive original bytes and command/stdout/environment receipts permanently."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
TEMP = ROOT / "outputs/reviewer-tools/phase4-2_4-factual-independent"
files = {
    "section.md": "section.md",
    "fence-1.py": "original.py",
    "bootstrap.py": "bootstrap.py",
    "extraction.json": "extraction.json",
    "execution.json": "original-execution.json",
    "environment.json": "original-environment.json",
    "stdout.txt": "original-stdout.txt",
    "stderr.txt": "original-stderr.txt",
    "figures/course/figures/rewrite-02-nonlinearity.svg": "nonlinearity.svg",
}
for source, name in files.items():
    (OUT / ("phase4-2_4-" + name)).write_bytes((TEMP / source).read_bytes())

env = dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
commands = [
    ("verification", [str(ROOT / ".venv/bin/python"), str(OUT / "phase4-2_4-verify.py")]),
    ("inkscape-version", ["inkscape", "--version"]),
    ("render", ["inkscape", "course/figures/rewrite-02-nonlinearity.svg", "--export-type=png", "--export-filename=docs/technical-reviews/artifacts/phase4-2_4-nonlinearity.png"]),
]
receipts = []
for name, argv in commands:
    stdout_path = OUT / f"phase4-2_4-{name}-stdout.txt"
    stderr_path = OUT / f"phase4-2_4-{name}-stderr.txt"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=45, check=False)
    receipts.append({
        "name": name, "argv": argv, "cwd": str(ROOT),
        "accessed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "exit_code": result.returncode,
        "stdout": {"path": stdout_path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(stdout_path.read_bytes()).hexdigest()},
        "stderr": {"path": stderr_path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest()},
        "environment": {"python": sys.version, "device_requested": "cpu", "CUDA_VISIBLE_DEVICES": "empty", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    })
    print(f"{name}: exit {result.returncode}, stdout {stdout_path.name}, stderr {stderr_path.name}")
    if result.returncode:
        raise SystemExit(result.returncode)
(OUT / "phase4-2_4-run-receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
