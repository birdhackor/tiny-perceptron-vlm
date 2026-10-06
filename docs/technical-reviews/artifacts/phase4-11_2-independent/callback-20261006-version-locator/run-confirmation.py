"""Capture the actual scope confirmation and single-section checker."""
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
command = [str(ROOT / ".venv/bin/python"), str(HERE / "confirm-typed-scope.py")]
started = time.perf_counter()
with (HERE / "inspection-stdout.txt").open("wb") as stdout, (HERE / "inspection-stderr.txt").open("wb") as stderr:
    result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=15, check=False)
checker = [str(ROOT / ".venv/bin/python"), "scripts/check_technical_reviews.py", "--lesson", "11.2"]
if result.returncode == 0:
    checked = subprocess.run(checker, cwd=ROOT, capture_output=True, text=True, timeout=15, check=False)
    (HERE / "checker-stdout.txt").write_text(checked.stdout)
    (HERE / "checker-stderr.txt").write_text(checked.stderr)
else:
    checked = None
receipt = {"checked_at": datetime.now(UTC).isoformat(), "confirmation_id": "callback-20261006-version-locator-actual-inspection",
    "command_argv": command, "command": shlex.join(command), "cwd": str(ROOT), "exit_code": result.returncode,
    "elapsed_seconds": time.perf_counter() - started,
    "environment": json.loads((HERE / "environment.json").read_bytes()) if (HERE / "environment.json").exists() else {},
    "canonical_report_path": "docs/technical-reviews/11.2.json", "canonical_report_sha256": sha(ROOT / "docs/technical-reviews/11.2.json"),
    "canonical_unchanged": sha(ROOT / "docs/technical-reviews/11.2.json") == "b76072807eab2587026b87aa6752b1f74a6dbc265dbe23d80aacf316b02047ca",
    "checker": {"command_argv": checker, "exit_code": checked.returncode, "stdout": checked.stdout, "stderr": checked.stderr,
        "program_sha256": sha(ROOT / "scripts/check_technical_reviews.py")} if checked else {"executed": False},
    "files": {path.name: sha(path) for path in sorted(HERE.iterdir()) if path.is_file() and path.name != "confirmation-receipt.json"},
    "scope": "Typed bounded-slice/full-section fingerprint confirmation only; no scientific claims or canonical report altered; prior complete PASS/proof/history byte-preserved."}
(HERE / "confirmation-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"inspection_exit": result.returncode, "checker_exit": checked.returncode if checked else None,
    "canonical_unchanged": receipt["canonical_unchanged"], "canonical_report_sha256": receipt["canonical_report_sha256"],
    "receipt_path": (HERE / "confirmation-receipt.json").relative_to(ROOT).as_posix(), "receipt_sha256": sha(HERE / "confirmation-receipt.json")}, ensure_ascii=False))
raise SystemExit(result.returncode or (checked.returncode if checked else 1))
