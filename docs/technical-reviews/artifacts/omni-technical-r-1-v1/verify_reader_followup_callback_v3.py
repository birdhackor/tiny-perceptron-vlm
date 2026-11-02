"""Supplement the saved build-era audit with the newly supplied intake v8."""

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
REVIEWS = ROOT / "docs/course-revision-20261029-omni/reviews"
EVIDENCE = ROOT / "docs/course-revision-20261029-omni/evidence"


def reference(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path.read_bytes()).hexdigest()}


intake_path = REVIEWS / "reader-late-callback-intake-v8.json"
intake = json.loads(intake_path.read_text())
entry = next(e for e in intake["entries"] if e["kind"] == "reader-current-v1")
current_path = ROOT / entry["path"]
assert reference(current_path)["sha256"] == entry["sha256"]
current = json.loads(current_path.read_text())
freeze_path = REVIEWS / "source-freeze-technical-fix13.json"
frozen = {p["page_id"]: p for p in json.loads(freeze_path.read_text())["pages"]}
memberships = []
for index, page in enumerate(current["pages"]):
    pid = page["page_id"]
    assert pid in ["9.12", "9.15"] and page["source_sha256"] == frozen[pid]["source_sha256"]
    assert page["figures_sha256"] == frozen[pid]["figures_sha256"]
    for ref in [page["basis_report"], page["session_evidence"]["opening"], *page["session_evidence"]["notes"]]:
        assert reference(ROOT / ref["path"])["sha256"] == ref["sha256"]
    memberships.append({"page_id": pid, "json_pointer": f"/pages/{index}", "source_sha256": page["source_sha256"], "basis_report": page["basis_report"], "basis_json_pointer": page["basis_json_pointer"], "session": page["session"], "notes_hash_checked": len(page["session_evidence"]["notes"]), "actual_full_raw_note_read": page["session_evidence"]["notes"][1]})

metadata_path = ROOT / "course/reading-times.json"
metadata = json.loads(metadata_path.read_text())
current_estimates, stale = [], []
for item in metadata["pages"]:
    page = frozen[item["page_id"]]
    target = current_estimates if item["source_sha256"] == page["source_sha256"] and item["figures_sha256"] == page["figures_sha256"] else stale
    target.append(item["page_id"])
assert len(current_estimates) == 368 and stale == ["first-steps"]
site_info = json.loads((ROOT / "outputs/omni-review-site-technical-fix13/build-info.json").read_text())
assert site_info["reading_time_pages"] == 366

destination = ROOT / "outputs/omni-technical-r-1-v3-negative-formal-current-metadata"
assert not destination.exists()
command = ["uv", "run", "--extra", "cpu", "python", "scripts/export_course.py", "--output", str(destination.relative_to(ROOT)), "--executed", "outputs/omni-notebooks-technical-fix13-current", "--reading-times", "course/reading-times.json", "--revision", "80c5db17e8b077b2c0a5e4e9baa48e760ccb0532", "--require-reading-times"]
process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
assert process.returncode != 0 and "first-steps" in process.stderr and not destination.exists()

result = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": "/root/omni_technical_coordinator/technical_r_1",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-1-v1/verify_reader_followup_callback_v3.py",
    "environment": {"python": sys.version, "device": "CPU; reader/source/estimate checks and negative export only"},
    "intake": reference(intake_path), "shared_current_report_collected_once": reference(current_path), "page_memberships": memberships,
    "current_reader_mapping": {"retained_exact_sources": 359, "earlier_actual_overlays": 3, "new_page_memberships_in_one_report": 2, "current_exact_source_total": 364, "pending_full_reader": ["first-steps"], "author_patch_without_new_reader": ["1.11"], "preserved_inherit": ["environment", "asset-storage", "publishing"]},
    "live_metadata": {**reference(metadata_path), "row_count": len(metadata["pages"]), "current_source_figure_rows": len(current_estimates), "stale_rows": stale, "merge_receipt": reference(EVIDENCE / "reading-times-late-callback-applied-v4.json")},
    "saved_site_build_info": site_info,
    "negative_formal_with_current_metadata": {"command": command, "exit_code": process.returncode, "stdout": process.stdout, "stderr": process.stderr, "output_created": destination.exists()},
    "scope": "The previously saved audit used the supplied older catalogue plus v5-v7 and the site's 366-row build input; it is not proof that v8 did not yet exist. This follow-up actually checks the newly supplied one-report/two-page intake and the latest 368 current-source AI estimates. Old raw evidence and the 366-row actual site build remain unchanged. This is not a new blind or human reading claim, publication, or cloud execution."}
with Path(__file__).with_name("reader-followup-callback-v3.json").open("x") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"shared_report": reference(current_path), "reader_current_source_total": 364, "live_current_AI_estimates": 368, "saved_site_AI_estimates": 366, "only_pending": stale, "formal_exit": process.returncode}, ensure_ascii=False, indent=2))
