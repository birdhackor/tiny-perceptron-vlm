"""Run the required per-section checker and retain an actual receipt."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[3]
checker = REPO / "scripts/check_technical_reviews.py"
report = REPO / "docs/technical-reviews/5.17.json"
argv = [str(REPO / ".venv/bin/python"), str(checker), "--lesson", "5.17"]
started = datetime.now(timezone.utc).isoformat()
completed = subprocess.run(argv, cwd=REPO, capture_output=True, check=False, timeout=60)
(BASE / "checker.stdout.txt").write_bytes(completed.stdout)
(BASE / "checker.stderr.txt").write_bytes(completed.stderr)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {"command_argv": argv, "cwd": str(REPO), "started_utc": started,
    "exit_code": completed.returncode, "timeout_seconds": 60,
    "stdout": completed.stdout.decode(), "stderr": completed.stderr.decode(),
    "stdout_sha256": sha(BASE / "checker.stdout.txt"),
    "stderr_sha256": sha(BASE / "checker.stderr.txt"),
    "checker_sha256": sha(checker), "report_sha256": sha(report),
    "python": sys.version,
    "scope": "Schema/byte-version/uniqueness gate for 5.17 only; independent factual judgment is in the fresh report."}
(BASE / "checker-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
raise SystemExit(completed.returncode)
