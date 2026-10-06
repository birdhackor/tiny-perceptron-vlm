"""Accept actual original continuity evidence; do not generate reader judgments."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.export_course import DOCUMENTS, home_introduction
from scripts.reading_time import build_inventory

root = Path.cwd()
base = root / "docs/course-revision-20261005/continuity"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_file(row):
    path = root / row["path"]
    assert path.is_file() and path.resolve().is_relative_to(root)
    assert sha(path) == row["sha256"], row["path"]
    if "bytes" in row:
        assert path.stat().st_size == row["bytes"]


inventory_path = base / "revised-04/inventory.json"
inventory = json.loads(inventory_path.read_text())
current = {p["page_id"]: p for p in inventory["pages"]}
actual = build_inventory(root, json.loads((root / "course/lesson-index.json").read_text()), DOCUMENTS,
                         home_introduction(json.loads((root / "course/lesson-index.json").read_text()), True))
assert len(current) == len(actual["pages"]) == 321
for page in actual["pages"]:
    saved = current[page["page_id"]]
    assert page["source_sha256"] == saved["source_sha256"]
    assert page["figures_sha256"] == saved["figures_sha256"]
    assert sha(root / saved["snapshot"]) == saved["source_sha256"]
    for name, digest in saved["figures_sha256"].items():
        assert sha(root / name) == digest

report_path = base / "reports/13_16-recheck-03.json"
trace_path = base / "traces/13_16-recheck-03.jsonl"
assert sha(report_path) == "a2bac90b1fbe8f7ef65687d60a992614af919fd23af9f16168b070ed4a7efc65"
assert sha(trace_path) == "f0bc1c7f37218884e657917e883815ad1e3c6abe6374d32f5420ed27a706f672"
report = json.loads(report_path.read_text())
assert report["reviewer_task"] == "/root/continuity_13_16"
assert report["current_verdict"] == "pass" and not report["must_fix"]
assert report["source_inventory_sha256"] == sha(inventory_path)
assert report["actual_reread_page_ids"] == ["13.11", "13.12", "13.13", "13.14", "13.15"]
assert report["changed_primary_page_ids"] == ["13.12", "13.14"]
assert report["same_exact_version_without_new_read_primary_count"] == 53
progress_path = base / "progress.json"
progress = json.loads(progress_path.read_text())
assignment = progress["assignments"]["13_16"]
rows = report["primary_page_results"]
assert len(rows) == len(assignment["primary_page_ids"]) == 58
assert {r["page_id"] for r in rows} == set(assignment["primary_page_ids"])
for row in rows:
    page = current[row["page_id"]]
    assert row["decision"] == "pass"
    assert row["source_sha256"] == page["source_sha256"]
    assert row["figures_sha256"] == page["figures_sha256"]
    assert sha(root / row["source_snapshot"]) == page["source_sha256"]

for key in ["prior_bytes_preservation"]:
    item = report[key]
    manifest = root / item["manifest_path"]
    assert sha(manifest) == item["manifest_sha256"]
    saved = json.loads(manifest.read_text())
    assert len(saved["files"]) == 149
    for row in saved["files"]:
        verify_file(row)
manifest = root / report["artifact_manifest_path"]
assert sha(manifest) == report["artifact_manifest_sha256"]
artifacts = json.loads(manifest.read_text())
assert artifacts["file_count"] == len(artifacts["files"]) == 19
for row in artifacts["files"]:
    verify_file(row)

gate_path = root / "docs/course-revision-20261005/verification/original-callbacks-after-continuity/gate-receipt.json"
gate = json.loads(gate_path.read_text())
assert gate["status"] == "passed" and len(gate["commands"]) == 5
assert all(c["exit_code"] == 0 for c in gate["commands"])
assert gate["current_dependency_scan"]["exit_code"] == 0
for item in gate["inputs"].values():
    verify_file(item)
for item in [*gate["commands"], gate["current_dependency_scan"]]:
    for prefix in ["stdout", "stderr"]:
        assert sha(root / item[prefix + "_path"]) == item[prefix + "_sha256"]

receipt = {
    "recorded_at": datetime.now(UTC).isoformat(), "group": "13_16",
    "report_path": str(report_path.relative_to(root)), "report_sha256": sha(report_path),
    "trace_path": str(trace_path.relative_to(root)), "trace_sha256": sha(trace_path),
    "primary_current_versions": 58, "actual_changed_primary_reread": 2,
    "actual_necessary_primary_reread": 3, "unchanged_prior_pass_inherited": 53,
    "initial_and_prior_rechecks_preserved_files": 149,
    "scope": "Root version/byte receipt only. SAME original third reader actually read five pages and viewed the changed diagram in desktop/mobile article context;53 identical-version own prior PASS judgments inherited. This is not a fresh full58 read or root scientific judgment.",
}
destination = base / "receipts/13_16-recheck-03.json"
assert not destination.exists()
destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
assignment.update(status="current_continuity_pass", current_report_path=receipt["report_path"],
                  current_report_sha256=receipt["report_sha256"], current_receipt_path=str(destination.relative_to(root)))
assignment["rechecks"][-1]["status"] = "current_original_recheck_pass_received"
assert all(a["status"] == "current_continuity_pass" for a in progress["assignments"].values())
assert sum(len(a["primary_page_ids"]) for a in progress["assignments"].values()) == 321
progress.update(status="phase4_current_continuity_and_original_callbacks_complete", completed_at=datetime.now(UTC).isoformat())
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n")
closure = {
    "recorded_at": datetime.now(UTC).isoformat(), "status": "phase4_complete",
    "current_inventory_path": str(inventory_path.relative_to(root)), "current_inventory_sha256": sha(inventory_path),
    "continuity_public_pages": 321, "original_factual_sections": 309,
    "current_original_technical_callback_sections": 75, "current_original_full_reader_callback_sections": 49,
    "default_metadata_gate_count": 5, "default_metadata_gate_exit_codes": [c["exit_code"] for c in gate["commands"]],
    "current_dependency_scan_exit_code": 0, "gate_receipt_sha256": sha(gate_path),
    "latest_continuity_receipt_path": str(destination.relative_to(root)), "latest_continuity_receipt_sha256": sha(destination),
    "acceptance_program_sha256": sha(Path(__file__)),
    "limitations": "Scientific/readability/continuity judgments belong to actual reviewers.11 public operations/home pages outside original309 factual scope remain explicitly scheduled for Phase6 after capstone engineering. No new GPU training in this phase; wording-only corrections reuse actual identical-code outputs.",
}
(root / "docs/course-revision-20261005/verification/phase4-closure.json").write_text(json.dumps(closure, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(closure, ensure_ascii=False))
