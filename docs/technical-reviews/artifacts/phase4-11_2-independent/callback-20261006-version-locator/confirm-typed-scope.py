"""Same-owner typed-scope confirmation; canonical report is never written."""
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
EXPECTED_REPORT_SHA = "b76072807eab2587026b87aa6752b1f74a6dbc265dbe23d80aacf316b02047ca"
def sha(raw): return hashlib.sha256(raw).hexdigest()
report_path = ROOT / "docs/technical-reviews/11.2.json"
report_bytes_before = report_path.read_bytes()
preserved_path = HERE / "prior-report.opaque.json"
assert report_bytes_before == preserved_path.read_bytes()
assert sha(report_bytes_before) == EXPECTED_REPORT_SHA
report = json.loads(report_bytes_before)
assert report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_11_2"
scope = report["read_scope"]["current_context_callback"]["necessary_context"]
assert scope["source"] == "course/chapters/05.md#5.1"
raw_5 = (ROOT / "course/chapters/05.md").read_bytes()
headings = list(re.finditer(rb"(?m)^## ([\w.]+) [^\r\n]+", raw_5))
i = next(i for i, item in enumerate(headings) if item[1] == b"5.1")
start, end = headings[i].start(), headings[i + 1].start()
section = raw_5[start:end]
marker = section.find(b"<details>")
assert marker >= 0
bounded = section[:marker]
assert sha(bounded) == scope["sha256"]
assert sha(section) == scope["full_section_sha256"]
assert scope["raw_byte_start"] == start and scope["raw_byte_end_exclusive"] == start + marker
assert scope["first_line"] == raw_5[:start].count(b"\n") + 1
assert scope["end_line"] == raw_5[:start + marker].count(b"\n")
old_bounded_snapshot = ROOT / scope["snapshot"]
current_bounded_snapshot = HERE / "current-5.1-inspected-slice.md"
assert old_bounded_snapshot.read_bytes() == current_bounded_snapshot.read_bytes() == bounded
assert section.startswith(bounded)
assert len(bounded) == 2646 and len(section) == 3467
previous_inspection = ROOT / "docs/technical-reviews/artifacts/phase4-11_2-independent/callback-20261006/inspection-results.json"
previous_result = json.loads(previous_inspection.read_bytes())
previous_locator = previous_result["current_context_locators"][1]
assert previous_locator == scope
# Fingerprint the primary section only: the unchanged primary was already read by this owner.
raw_11 = (ROOT / "course/chapters/11.md").read_bytes()
h11 = list(re.finditer(rb"(?m)^## ([\w.]+) [^\r\n]+", raw_11))
j = next(j for j, item in enumerate(h11) if item[1] == b"11.2")
primary = raw_11[h11[j].start():h11[j + 1].start()]
assert sha(primary) == report["source_sha256"] == "5b6d0b4819041b601df9a10e2a535477096af071d2d2122d93c4251520e0e019"
assert primary == (ROOT / "docs/technical-reviews/artifacts/phase4-11_2-independent/section.md").read_bytes()
environment = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
    "execution_kind": "raw byte inclusion/range/SHA and same-owner scope metadata inspection; no model imports or execution"}
result = {"checked_at": datetime.now(UTC).isoformat(), "reviewer_task": report["reviewer_task"],
    "canonical_report": {"path": report_path.relative_to(ROOT).as_posix(), "sha256": sha(report_bytes_before), "unchanged": True},
    "prior_opaque": {"path": preserved_path.relative_to(ROOT).as_posix(), "sha256": sha(preserved_path.read_bytes())},
    "primary_source_sha256": sha(primary),
    "inspected_metadata_pointer": "/read_scope/current_context_callback/necessary_context",
    "metadata": scope,
    "measurements": {"current_full_section_bytes": len(section), "current_full_section_sha256": sha(section),
        "current_bounded_slice_bytes": len(bounded), "current_bounded_slice_sha256": sha(bounded),
        "slice_byte_start": start, "slice_byte_end_exclusive": start + marker,
        "bounded_is_exact_prefix_of_current_full_section": True,
        "previous_saved_snapshot_equals_current_bounded_slice": True,
        "previous_actual_inspection_locator_equals_canonical_locator": True},
    "actual_read_scope": "Current 5.1 main text, Python fence and exercise before details, source lines5–40; the complete 5.1 section was fingerprinted mechanically without claiming the details/history were read or reviewed. Primary11.2 only fingerprinted against already-read original raw in this confirmation.",
    "field_meaning_and_time": "The canonical necessary_context sha256 is explicitly the raw inspected slice, governed by inspection_scope, raw_byte_start/end, first/end_line and snapshot. full_section_sha256 separately identifies the encompassing full current 5.1 section. Both were recorded by the same owner's prior 2026-10-06 context callback and both still match current bytes. This is neither an obsolete full-section fingerprint nor a new scientific discrepancy.",
    "support_boundary": "The necessary 5.1 main slice still supports the 11.2 distinction among gradient, optimizer membership and step. Existing original authority/code/CPU proofs remain under their original exact-hash and historical execution declarations; this confirmation does not alter claims, dates or findings and does not rerun them.",
    "decision": "Canonical metadata was already explicitly bounded and correct; preserve complete existing canonical report/proof/history without editing. This confirmation is a typed-scope/version check, not a new scientific review.",
    "environment": environment,
    "previous_actual_inspection_sha256": sha(previous_inspection.read_bytes()),
    "limits": "No model import, forward, backward, scalar step, training, GPU, model/paper download, source modification, commit or other report read. No diagrams or visual claims need rendering for this typed-scope confirmation."}
assert report_path.read_bytes() == report_bytes_before
(HERE / "typed-scope-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
(HERE / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
