"""Run this section's actual schema/version checker and save its receipt."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
command = [str(ROOT / ".venv/bin/python"), "scripts/check_technical_reviews.py", "--lesson", "6.7"]
result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
(BASE / "checker-stdout.txt").write_text(result.stdout, encoding="utf-8")
(BASE / "checker-stderr.txt").write_text(result.stderr, encoding="utf-8")
report = ROOT / "docs/technical-reviews/6.7.json"
receipt = {"command_argv": command, "cwd": str(ROOT), "exit_code": result.returncode,
           "checked_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
           "scope": "Only --lesson 6.7; checker reads ownership metadata automatically, no other review conclusions inspected by reviewer.",
           "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
           "checker_sha256": hashlib.sha256((ROOT / "scripts/check_technical_reviews.py").read_bytes()).hexdigest(),
           "stdout": result.stdout, "stderr": result.stderr}
(BASE / "checker-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
raise SystemExit(result.returncode)
