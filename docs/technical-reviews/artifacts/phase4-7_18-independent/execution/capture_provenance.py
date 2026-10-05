"""Save actual bounded-check/render commands, outputs, hashes, and environment."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[5]
base = Path(__file__).resolve().parents[1]
commands = [
    ("bounded", [str(repo / ".venv/bin/python"), str(base / "execution/bounded_checks.py")]),
    ("render", ["inkscape", "course/figures/rewrite-07-18-feedback-materials.svg", "--export-type=png", "--export-filename=" + str(base / "figures/feedback-materials.png"), "--export-width=640"]),
]
records = []
for name, argv in commands:
    completed = subprocess.run(argv, cwd=repo, capture_output=True, timeout=30)
    stdout = base / "execution" / (name + ".stdout.txt")
    stderr = base / "execution" / (name + ".stderr.txt")
    stdout.write_bytes(completed.stdout)
    stderr.write_bytes(completed.stderr)
    records.append({"name": name, "argv": argv, "cwd": str(repo), "timeout_seconds": 30, "exit_code": completed.returncode,
                    "outputs": [{"path": str(p.relative_to(repo)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (stdout, stderr)]})
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
versions = {}
for name, argv in [("inkscape", ["inkscape", "--version"]), ("pdftotext", ["pdftotext", "-v"])]:
    completed = subprocess.run(argv, capture_output=True, timeout=10)
    versions[name] = (completed.stdout + completed.stderr).decode().strip()
record = {"python": sys.version, "device": "CPU", "versions": versions, "commands": records}
(base / "execution/command-provenance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(record, ensure_ascii=False, indent=2))
