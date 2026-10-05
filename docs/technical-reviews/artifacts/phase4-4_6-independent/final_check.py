"""Run the selected checker once and retain actual status/provenance."""
import hashlib
import json
import subprocess
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
command = [str(ROOT / ".venv/bin/python"), "scripts/check_technical_reviews.py", "--lesson", "4.6"]
result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30, check=False)
(ART / "checker.stdout.txt").write_bytes(result.stdout)
(ART / "checker.stderr.txt").write_bytes(result.stderr)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
(ART / "checker-receipt.json").write_text(json.dumps({"command_argv": command, "cwd": str(ROOT),
    "timeout_seconds": 30, "exit_code": result.returncode,
    "checker_sha256": sha(ROOT / "scripts/check_technical_reviews.py"),
    "report_sha256": sha(ROOT / "docs/technical-reviews/4.6.json"),
    "stdout_sha256": sha(ART / "checker.stdout.txt"), "stderr_sha256": sha(ART / "checker.stderr.txt"),
    "scope": "Schema, independence IDs and current byte hashes only; checker is not a factual judgment."},
    ensure_ascii=False, indent=2) + "\n")
print(result.stdout.decode(), end="")
print(result.stderr.decode(), end="")
raise SystemExit(result.returncode)
