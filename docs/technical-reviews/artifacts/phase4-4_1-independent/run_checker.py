from pathlib import Path
import hashlib
import json
import shlex
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
command = [str(ROOT / ".venv/bin/python"), "scripts/check_technical_reviews.py", "--lesson", "4.1"]
run = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=45)
receipt = {"command": shlex.join(command), "cwd": str(ROOT), "exit_code": run.returncode,
           "stdout": run.stdout.decode(), "stderr": run.stderr.decode(),
           "checker_sha256": hashlib.sha256((ROOT / "scripts/check_technical_reviews.py").read_bytes()).hexdigest(),
           "report_sha256": hashlib.sha256((ROOT / "docs/technical-reviews/4.1.json").read_bytes()).hexdigest()}
(OUT / "checker-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
raise SystemExit(run.returncode)
