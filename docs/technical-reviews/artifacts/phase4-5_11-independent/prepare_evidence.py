"""Freeze the section and small original inputs; execute its exact fence on CPU."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

paths = [
    "docs/review-tools/factual-reviewer-instructions.md",
    "docs/review-tools/section_facts.py",
    ".agents/skills/clear-tutorial/references/review-protocol.md",
    "scripts/check_technical_reviews.py", "scripts/build_course.py",
    "tiny_perceptron/data.py", "tiny_perceptron/__init__.py",
    "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
    "scripts/course_experiments/run.py", "pyproject.toml",
    "docs/course-experiments/results/real_text.json", "assets/training/manifest.json",
    "data/training/text-initial/tinystories-train-512.jsonl",
    "data/training/text-initial/chinese-classical-train-365.jsonl",
]
provenance = []
for rel in paths:
    source = ROOT / rel
    dest = OUT / "inputs" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    provenance.append({"original_path": rel, "snapshot": str(dest.relative_to(ROOT)),
                       "sha256": sha(dest), "bytes": dest.stat().st_size})
run = Path("/tmp/phase4-5_11-independent-original")
command = [str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py",
           "course/chapters/05.md#5.11", "--execute", "--timeout", "45", "--output", str(run)]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=55)
(OUT / "prepare.stdout.txt").write_bytes(completed.stdout)
(OUT / "prepare.stderr.txt").write_bytes(completed.stderr)
(OUT / "prepare-receipt.json").write_text(json.dumps({"command_argv": command,
    "cwd": str(ROOT), "exit_code": completed.returncode, "timeout_seconds":55},indent=2)+"\n")
if completed.returncode:
    raise SystemExit(completed.returncode)
for p in run.iterdir():
    if p.is_file():
        (OUT / "original").mkdir(exist_ok=True)
        shutil.copyfile(p, OUT / "original" / p.name)
extraction = json.loads((OUT / "original/extraction.json").read_text())
provenance.append({"original_path":"course/chapters/05.md#5.11",
    "snapshot": str((OUT / "original/section.md").relative_to(ROOT)),
    "sha256":extraction["source_sha256"], "bytes":(OUT/"original/section.md").stat().st_size})
(OUT / "input-provenance.json").write_text(json.dumps({"policy":
    "Small source/data inputs copied as original bytes. No model weights used or copied.",
    "inputs":provenance},ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"exit_code":completed.returncode,"source_sha256":extraction["source_sha256"],
    "svg_references":extraction["svg_references"],"fences":extraction["python_fences"]},ensure_ascii=False,indent=2))
