"""Execute the single-section schema/hash checker and retain a truthful receipt."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
command = [str(ROOT / ".venv/bin/python"), "scripts/check_technical_reviews.py", "--lesson", "7.6"]
result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
(BASE / "checker.stdout.txt").write_bytes(result.stdout)
(BASE / "checker.stderr.txt").write_bytes(result.stderr)
report = ROOT / "docs/technical-reviews/7.6.json"
receipt = {"command_argv": command, "cwd": str(ROOT), "exit_code": result.returncode,
           "stdout_file": "checker.stdout.txt", "stderr_file": "checker.stderr.txt",
           "python": sys.version, "platform": platform.platform(),
           "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
           "checker_sha256": hashlib.sha256((ROOT / "scripts/check_technical_reviews.py").read_bytes()).hexdigest(),
           "source_sha256": json.loads(report.read_text())["source_sha256"],
           "scope": "Only --lesson 7.6; validates schema/evidence hashes and independence identifiers, not technical truth"}
(BASE / "checker-receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(receipt, indent=2, ensure_ascii=False))
print(result.stdout.decode(), end="")
raise SystemExit(result.returncode)
