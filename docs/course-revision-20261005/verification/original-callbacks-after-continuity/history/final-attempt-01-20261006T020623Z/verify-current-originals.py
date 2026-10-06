"""Run current administrative gates only after genuine original callbacks close."""

from pathlib import Path
import datetime
import hashlib
import json
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
inputs = {
    "factual_progress": ROOT / "docs/course-revision-20261005/factual-review-progress.json",
    "reader_progress": ROOT / "docs/course-revision-20261005/review-progress.json",
    "original_callback_ledger": ROOT / "docs/course-revision-20261005/continuity/original-review-callbacks.json",
}
raw = {name: path.read_bytes() for name, path in inputs.items()}
p, r, ledger = (json.loads(raw[name]) for name in inputs)
ids = p["section_order"]
assert len(ids) == len(set(ids)) == 309
assert all(
    p["records"][sid]["status"] == "pass"
    and p["records"][sid]["report_version_current"]
    and p["records"][sid]["checker_exit_code"] == 0
    and not p["records"][sid].get("dependency_recheck_pending", False)
    and not p["records"][sid].get("continuity_repair_callback_pending", False)
    for sid in ids
)
assert all(
    r["records"][sid]["status"] == "pass"
    and not r["records"][sid]["format_validation_issues"]
    for sid in ids
)
assert all(rec["current_complete"] for rec in ledger["records"].values())
assert all(
    rec["technical"]["status"] == "pass"
    and rec["technical"]["actual_dispatch_method"] == "collaboration.followup_task"
    and rec["technical"]["agent_completion_received_or_status_observed"]
    for rec in ledger["records"].values()
)
required_readers = [
    rec["reader"] for rec in ledger["records"].values()
    if rec["reader"]["status"] != "not_required_source_unchanged"
]
assert all(
    role["status"] == "pass"
    and role["actual_dispatch_method"] == "collaboration.followup_task"
    and role["agent_completion_received_or_status_observed"]
    for role in required_readers
)
receipt = {
    "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "scope": "Current schema, source/evidence bytes, original task callback receipts and incremental trace metadata. These gates do not perform scientific claim review or prove firsthand inspection; original owners performed and reported those separately. The 309-section factual scope does not include all 321 public pages.",
    "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
    "inputs": {name: {"path": str(inputs[name].relative_to(ROOT)), "sha256": sha(data)} for name, data in raw.items()},
    "current_section_count": len(ids),
    "same_original_technical_callbacks": len(ledger["records"]),
    "same_original_full_reader_callbacks": len(required_readers),
    "callback_count_basis": "The two preceding counts are distinct sections with a closed current original-owner role, not the number of followup tool calls or historical sessions. Genuine repeated dispatches, revisions and FULL sessions remain in the callback ledger and prior role/trace histories.",
    "commands": [],
    "status": "running",
}
commands = [
    ("technical-schema", [sys.executable, "scripts/check_technical_reviews.py"]),
    ("reader-schema", [sys.executable, "scripts/check_course_reviews.py"]),
    ("whole-round-version", [sys.executable, "docs/review-tools/check_review_round.py", "--stage", "all", "--output", str(OUT / "whole-round-version.json")]),
    ("factual-dispatch-artifact-version", [sys.executable, "docs/review-tools/audit_factual_round.py", "--output", str(OUT / "factual-dispatch-artifact-version.json")]),
    ("reader-incremental-trace-version", [sys.executable, "docs/review-tools/audit_readability_round.py", "--output", str(OUT / "reader-incremental-trace-version.json")]),
]
for name, command in commands:
    started = datetime.datetime.now(datetime.UTC).isoformat()
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    stdout, stderr = OUT / (name + ".stdout.txt"), OUT / (name + ".stderr.txt")
    stdout.write_bytes(result.stdout)
    stderr.write_bytes(result.stderr)
    receipt["commands"].append({
        "name": name, "command": command, "cwd": str(ROOT), "started_at": started,
        "completed_at": datetime.datetime.now(datetime.UTC).isoformat(), "exit_code": result.returncode,
        "stdout_path": str(stdout.relative_to(ROOT)), "stdout_sha256": sha(result.stdout),
        "stderr_path": str(stderr.relative_to(ROOT)), "stderr_sha256": sha(result.stderr),
    })
    receipt["status"] = "failed" if result.returncode else "running"
    (OUT / "gate-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"gate": name, "exit_code": result.returncode}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
dependency_command = [sys.executable, str(OUT / "scan-current-dependencies.py")]
dependency_started = datetime.datetime.now(datetime.UTC).isoformat()
dependency_result = subprocess.run(dependency_command, cwd=ROOT, capture_output=True)
dependency_stdout = OUT / "current-dependency-scan.stdout.txt"
dependency_stderr = OUT / "current-dependency-scan.stderr.txt"
dependency_stdout.write_bytes(dependency_result.stdout)
dependency_stderr.write_bytes(dependency_result.stderr)
receipt["current_dependency_scan"] = {
    "command": dependency_command,
    "started_at": dependency_started,
    "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "exit_code": dependency_result.returncode,
    "stdout_path": str(dependency_stdout.relative_to(ROOT)),
    "stdout_sha256": sha(dependency_result.stdout),
    "stderr_path": str(dependency_stderr.relative_to(ROOT)),
    "stderr_sha256": sha(dependency_result.stderr),
}
if dependency_result.returncode:
    receipt["status"] = "failed_current_dependency_scan"
    (OUT / "gate-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"current_dependency_scan_exit_code": dependency_result.returncode}), flush=True)
    raise SystemExit(dependency_result.returncode)
assert all(inputs[name].read_bytes() == data for name, data in raw.items()), "Gate inputs changed during execution."
receipt.update(status="passed", completed_at=datetime.datetime.now(datetime.UTC).isoformat(), verification_program_sha256=sha(Path(__file__).read_bytes()))
(OUT / "gate-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "passed", "sections": len(ids), "gate_count": len(commands), "current_dependency_scan_exit_code": 0}))
