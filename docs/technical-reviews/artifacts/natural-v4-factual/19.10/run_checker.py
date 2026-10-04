from pathlib import Path
import hashlib
import json
import platform
import subprocess
import torch

base = Path("docs/technical-reviews/artifacts/natural-v4-factual/19.10")
report = Path("docs/technical-reviews/19.10.json")
first = base / "first-own-report.json"
if not first.exists():
    first.write_bytes(report.read_bytes())
command = [".venv/bin/python", "scripts/check_technical_reviews.py", "--lesson", "19.10"]
run = subprocess.run(command, capture_output=True, text=True)
base.joinpath("checker.stdout.txt").write_text(run.stdout)
base.joinpath("checker.stderr.txt").write_text(run.stderr)
receipt = {"command": command, "exit_code": run.returncode,
           "stdout_path": str(base / "checker.stdout.txt"), "stderr_path": str(base / "checker.stderr.txt"),
           "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
           "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
           "scope": "Actual unmodified official checker; checks provenance/schema/current files, not factual truth. Revise is retained for confirmed normalization contradiction."}
base.joinpath("checker-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
print(run.stdout, end="")
print(run.stderr, end="")
raise SystemExit(run.returncode)
