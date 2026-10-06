"""Register actual callback logs, then run only the current 9.8 formal checker."""

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
REL = BASE.relative_to(ROOT).as_posix()
REPORT = ROOT / "docs/technical-reviews/9.8.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


report = json.loads(REPORT.read_bytes())
stdout = (BASE / "callback.stdout.txt").read_bytes()
stderr = (BASE / "callback.stderr.txt").read_bytes()
assert stdout.endswith(b"CALLBACK_COMPLETE\n") and not stderr
for identifier, filename, kind, description in [
    ("callback-stdout", "callback.stdout.txt", "source_snapshot", "Actual callback.py stdout: raw diff and all successful fingerprint checks. Printed intermediate report hash predates registration of this log; final report hash is in formal-check-receipt.json."),
    ("callback-stderr", "callback.stderr.txt", "source_snapshot", "Actual empty callback.py stderr, exit0 confirmed by tool receipt."),
    ("callback-finalize-code", "finalize.py", "code", "Actual code registering completed stdout/stderr and invoking only the single-section formal checker after new report exists."),
]:
    assert identifier not in {a["id"] for a in report["artifacts"]}
    report["artifacts"].append({"id": identifier, "path": REL + "/" + filename,
                                "sha256": digest((BASE / filename).read_bytes()), "kind": kind,
                                "description": description})
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
command = [str(ROOT / ".venv/bin/python"), "scripts/check_technical_reviews.py", "--lesson", "9.8"]
completed = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
(BASE / "formal-check.stdout.txt").write_bytes(completed.stdout)
(BASE / "formal-check.stderr.txt").write_bytes(completed.stderr)
assert completed.returncode == 0, completed.stdout.decode() + completed.stderr.decode()
receipt = {
    "command": ".venv/bin/python scripts/check_technical_reviews.py --lesson 9.8", "exit_code": completed.returncode,
    "checked_at": datetime.datetime.now(datetime.UTC).isoformat(),
    "checker_sha256": digest((ROOT / "scripts/check_technical_reviews.py").read_bytes()),
    "report_path": "docs/technical-reviews/9.8.json", "report_sha256": digest(REPORT.read_bytes()),
    "source_sha256": report["source_sha256"], "verdict": report["verdict"],
    "prior_opaque_path": REL + "/prior-report.opaque.json",
    "prior_opaque_sha256": digest((BASE / "prior-report.opaque.json").read_bytes()),
    "canonical_prior_artifact_id": "callback-prior-report-opaque",
    "prior_present_in_current_report_artifacts": any(a["id"] == "callback-prior-report-opaque" and a["path"] == REL + "/prior-report.opaque.json" and a["sha256"] == digest((BASE / "prior-report.opaque.json").read_bytes()) for a in report["artifacts"]),
    "callback_command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-9_8-callback-20261006/callback.py",
    "callback_exit_code": 0, "callback_stdout_sha256": digest(stdout), "callback_stderr_sha256": digest(stderr),
    "actual_inspection_sha256": digest((BASE / "actual-inspection.json").read_bytes()),
    "stdout_sha256": digest(completed.stdout), "stderr_sha256": digest(completed.stderr),
    "result": completed.stdout.decode(),
    "scope": "Formal schema/hash check only. Own current fullsection/context/diff inspection identified only glyph edits; all unchanged evidence was reused after actual fingerprint checks. No new model score, source fetch, training or inference is claimed.",
}
assert receipt["prior_present_in_current_report_artifacts"] is True
(BASE / "formal-check-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"verdict": report["verdict"], "report_sha256": receipt["report_sha256"],
                  "prior_opaque_sha256": receipt["prior_opaque_sha256"],
                  "prior_artifact_id": receipt["canonical_prior_artifact_id"],
                  "receipt_path": REL + "/formal-check-receipt.json",
                  "receipt_sha256": digest((BASE / "formal-check-receipt.json").read_bytes()),
                  "checker_exit_code": completed.returncode}, ensure_ascii=False))
