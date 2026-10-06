"""Persist the observed chapter snapshot, then run only the current 7.8 checker."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[3]
OUT = BASE / "reinspection-20261006"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


report_path = REPO / "docs/technical-reviews/7.8.json"
report = json.loads(report_path.read_text())
receipt_path = OUT / "reinspection-receipt.json"
receipt = json.loads(receipt_path.read_text())
chapter_bytes = (REPO / "course/chapters/07.md").read_bytes()
chapter_digest = hashlib.sha256(chapter_bytes).hexdigest()
assert chapter_digest == receipt["source_file_frozen_input"]["observed_sha256"]
chapter_path = OUT / "chapter-frozen-at-reinspection.md"
chapter_path.write_bytes(chapter_bytes)
receipt["source_file_frozen_input"].update(snapshot_path=str(chapter_path.relative_to(REPO)), snapshot_sha256=digest(chapter_path),
                                           description="Complete raw chapter bytes frozen at this reinspection, hash unchanged when saved. Only 7.8 and necessary7.4/7.7 were semantically read; this is not a current-chapter assertion after future edits.")
save(receipt_path, receipt)
receipt_id = report["reinspection"]["artifact_id"]
for artifact in report["artifacts"]:
    if artifact["id"] == receipt_id:
        artifact["sha256"] = digest(receipt_path)
report["reinspection"]["receipt_sha256"] = digest(receipt_path)
report["artifacts"].append({"id": "chapter-frozen-at-reinspection-20261006", "path": str(chapter_path.relative_to(REPO)),
                            "sha256": digest(chapter_path), "kind": "source_snapshot",
                            "description": "本次observed完整chapter raw bytes之frozen版本；仅current7.8與必要7.4/7.7語义讀取，非未來整章指紋聲稱。"})
report["artifacts"].append({"id": "reinspection-checker-code-20261006", "path": str(Path(__file__).relative_to(REPO)),
                            "sha256": digest(Path(__file__)), "kind": "code",
                            "description": "本次保留raw章snapshot、更新本人的canonical receipt指紋並真正跑單節checker之實際腳本。"})
save(report_path, report)

for artifact in report["artifacts"]:
    assert digest(REPO / artifact["path"]) == artifact["sha256"], artifact["id"]
spec = importlib.util.spec_from_file_location("facts", REPO / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
current, _, _ = facts.original_section(REPO / "course/chapters/07.md", "7.8")
assert hashlib.sha256(current).hexdigest() == report["source_sha256"]
assert digest(REPO / report["reinspection"]["prior_report_history_path"]) == report["reinspection"]["prior_report_sha256"]
argv = [str(REPO / ".venv/bin/python"), "scripts/check_technical_reviews.py", "--lesson", "7.8"]
completed = subprocess.run(argv, cwd=REPO, capture_output=True, check=False, timeout=30)
(OUT / "checker.stdout.txt").write_bytes(completed.stdout)
(OUT / "checker.stderr.txt").write_bytes(completed.stderr)
checker_receipt = {"kind": "single_section_actual_checker_receipt", "command_argv": argv, "command": shlex.join(argv),
                   "cwd": str(REPO), "timeout_seconds": 30, "exit_code": completed.returncode,
                   "recorded_at": datetime.now(timezone.utc).isoformat(),
                   "environment": {"python": sys.version, "executable": sys.executable, "device": "CPU format/hash checker; no model computation"},
                   "report_sha256": digest(report_path), "source_sha256": report["source_sha256"],
                   "checker_sha256": digest(REPO / "scripts/check_technical_reviews.py"), "stdout": completed.stdout.decode(), "stderr": completed.stderr.decode(),
                   "stdout_sha256": digest(OUT / "checker.stdout.txt"), "stderr_sha256": digest(OUT / "checker.stderr.txt"),
                   "scope": "Only7.8 selected after owner's canonical update; metadata checker does not replace technical judgment."}
save(OUT / "checker-receipt.json", checker_receipt)
summary_path = OUT / "reinspection-summary.json"
summary = json.loads(summary_path.read_text())
summary.update(report_sha256=digest(report_path), receipt_sha256=digest(receipt_path),
               checker_receipt_path=str((OUT / "checker-receipt.json").relative_to(REPO)), checker_receipt_sha256=digest(OUT / "checker-receipt.json"), checker_exit_code=completed.returncode)
save(summary_path, summary)
save(OUT / "sha256-manifest.json", {"report_path": str(report_path.relative_to(REPO)), "report_sha256": digest(report_path),
                                  "files": [{"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "sha256-manifest.json"],
                                  "required_artifacts_outside_docs": [], "scope": "Reinspection proof only; original independent proof preserved unchanged."})
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(completed.returncode)
