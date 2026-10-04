from pathlib import Path
import hashlib
import json
import platform
import subprocess
import torch

out = Path("docs/technical-reviews/artifacts/natural-v4-factual/19.10/round2")
report = Path("docs/technical-reviews/19.10.json")
command = [".venv/bin/python", "scripts/check_technical_reviews.py", "--lesson", "19.10"]
run = subprocess.run(command, capture_output=True, text=True)
out.joinpath("checker.stdout.txt").write_text(run.stdout)
out.joinpath("checker.stderr.txt").write_text(run.stderr)
receipt = {"command": command, "exit_code": run.returncode,
           "stdout_path": str(out / "checker.stdout.txt"), "stderr_path": str(out / "checker.stderr.txt"),
           "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
           "source_sha256": json.loads(report.read_text())["source_sha256"],
           "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
           "scope": "Actual unmodified official checker after original-owner full recheck. Checks current report/provenance/evidence schema; factual decision rests on actual original-source inspection and bounded CPU proof. Original checker1 and first revise report remain unchanged outside round2."}
out.joinpath("checker-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
print(run.stdout, end="")
print(run.stderr, end="")
raise SystemExit(run.returncode)
