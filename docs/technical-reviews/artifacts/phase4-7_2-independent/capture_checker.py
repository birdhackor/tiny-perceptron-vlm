"""Persist the actual single-section checker receipt, without rewriting its verdict."""
import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
argv = [str(ROOT / ".venv/bin/python"), "scripts/check_technical_reviews.py", "--lesson", "7.2"]
proc = subprocess.run(argv, cwd=ROOT, capture_output=True, timeout=30)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
(BASE / "checker.stdout.txt").write_bytes(proc.stdout)
(BASE / "checker.stderr.txt").write_bytes(proc.stderr)
receipt = {"command_argv":argv,"cwd":str(ROOT),"timeout_seconds":30,"exit_code":proc.returncode,
    "report_sha256":sha(ROOT / "docs/technical-reviews/7.2.json"),
    "checker_sha256":sha(ROOT / "scripts/check_technical_reviews.py"),
    "stdout_sha256":sha(BASE / "checker.stdout.txt"),"stderr_sha256":sha(BASE / "checker.stderr.txt"),
    "environment":json.loads((BASE / "original-environment.json").read_text()),
    "meaning":"Checker verifies report schema, identity and referenced versions only; factual pass is the reviewer's original-source assessment."}
(BASE / "checker-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
manifest = json.loads((BASE / "manifest.json").read_text())
manifest["checker_receipt"] = {"path":str((BASE / "checker-receipt.json").relative_to(ROOT)),"sha256":sha(BASE / "checker-receipt.json"),"exit_code":proc.returncode}
(BASE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(proc.stdout.decode(), end="")
print(json.dumps({"checker_exit_code":proc.returncode,"report_sha256":receipt["report_sha256"]}))
raise SystemExit(proc.returncode)
