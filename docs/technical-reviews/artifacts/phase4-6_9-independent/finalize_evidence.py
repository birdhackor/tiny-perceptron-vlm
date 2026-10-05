"""Capture the actual per-section checker and audit permanent evidence identity."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
report_command = [str(ROOT / ".venv/bin/python"), str(OUT / "build_report.py")]
compiled = subprocess.run(report_command, cwd=ROOT, capture_output=True, timeout=30)
(OUT / "report-summary.json").write_bytes(compiled.stdout)
assert compiled.returncode == 0, compiled.stderr.decode()
command = [str(ROOT / ".venv/bin/python"), "scripts/check_technical_reviews.py", "--lesson", "6.9"]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30)
(OUT / "checker-stdout.txt").write_bytes(completed.stdout)
(OUT / "checker-stderr.txt").write_bytes(completed.stderr)
report_path = ROOT / "docs/technical-reviews/6.9.json"
report = json.loads(report_path.read_text())
artifacts = report["artifacts"]
assert all(sha(ROOT / artifact["path"]) == artifact["sha256"] for artifact in artifacts)
assert all(Path(artifact["path"]).is_relative_to("docs/technical-reviews/artifacts") for artifact in artifacts)
assert not any(Path(artifact["path"]).suffix == ".pt" for artifact in artifacts)
ignore = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT, input="\n".join(artifact["path"] for artifact in artifacts).encode(), capture_output=True)
assert ignore.returncode == 1 and not ignore.stdout, ignore.stdout.decode()
receipt = {"checked_at": datetime.now(timezone.utc).isoformat(), "command_argv": command, "cwd": str(ROOT), "exit_code": completed.returncode, "stdout_path": str((OUT / "checker-stdout.txt").relative_to(ROOT)), "stdout_sha256": sha(OUT / "checker-stdout.txt"), "stderr_path": str((OUT / "checker-stderr.txt").relative_to(ROOT)), "stderr_sha256": sha(OUT / "checker-stderr.txt"), "checker_sha256": sha(ROOT / "scripts/check_technical_reviews.py"), "report_path": "docs/technical-reviews/6.9.json", "report_sha256": sha(report_path), "source_sha256": report["source_sha256"], "verdict": report["verdict"], "registered_artifacts": len(artifacts), "all_artifact_hashes_match": True, "formal_artifacts_ignored_by_git": False, "model_weights_copied_or_loaded": False, "review_scope": "Only 6.9 independent correctness review; no course text/figures changed, no commit, no old review body or verdict inspected."}
write(OUT / "checker-receipt.json", receipt)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
raise SystemExit(completed.returncode)
