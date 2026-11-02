"""Bind the latest v5/fix07 freeze to the already inspected W.1 programs and new export."""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
F = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v5.json"
C = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix07.json"
R = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_w_1/technical-original-v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


started = datetime.now(UTC).isoformat()
original = json.loads(R.read_bytes())
freeze = json.loads(F.read_bytes())
canonical = json.loads(C.read_bytes())
assert sha(R) == "18525e63a0d34a51e9acc1eab7938842138b672ba70f33b03981ceb7cebe5175"
assert freeze["canonical_source_freeze"] == {"path": str(C.relative_to(ROOT)), "sha256": sha(C)}
page = next(p for p in freeze["pages"] if p["page_id"] == "W.1")
raw = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == page["source_sha256"] == original["source_sha256"]
assert page["figures_sha256"] == original["figure_sha256"] == {}
old = json.loads((ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v4.json").read_bytes())
old_page = next(p for p in old["pages"] if p["page_id"] == "W.1")
assert raw == (ROOT / old_page["snapshot"]).read_bytes()
parent = next(p for p in canonical["pages"] if p["page_id"] == "first-steps")
parent_raw = (ROOT / parent["snapshot"]).read_bytes()
assert hashlib.sha256(parent_raw).hexdigest() == parent["source_sha256"]
assert parent_raw == (ROOT / parent["source"]).read_bytes()
assert re.search(rb"^## W\.1 .*?(?=^## |\Z)", parent_raw, re.M | re.S)[0] == raw
dependencies = []
for source in original["sources"]:
    if source["kind"] == "repository_code":
        actual = sha(ROOT / source["path"])
        assert actual == source["sha256"], source["path"]
        dependencies.append({"path": source["path"], "sha256": actual})

exports_path = A / "isolated-export-current-v2.json"
exports = json.loads(exports_path.read_bytes())
assert exports["exit_code"] == 0 and exports["bundle_verified"] == exports["isolated_downloads_identical"] == 331
verified = {p["lesson_id"]: p for p in exports["bundle_records"]}
notebook_pages = [p for p in canonical["pages"] if p.get("notebook")]
assert len(notebook_pages) == len(verified) == 331
assert {p["page_id"] for p in notebook_pages} == set(verified)
isolated = Path(exports["isolated_root"])
for page_record in notebook_pages:
    digest = sha(ROOT / page_record["notebook"])
    assert digest == page_record["notebook_sha256"] == verified[page_record["page_id"]]["notebook_sha256"]
    assert (ROOT / page_record["notebook"]).read_bytes() == (isolated / page_record["notebook"]).read_bytes()
    assert (ROOT / page_record["notebook"]).read_bytes() == (isolated / "outputs/site" / page_record["notebook"]).read_bytes()
for relative in ["course/README.md", "course/first-steps.md", "scripts/build_course.py", "scripts/export_course.py"]:
    assert (ROOT / relative).read_bytes() == (isolated / relative).read_bytes(), relative

prior_path = A / "recheck-current-v2.json"
prior = json.loads(prior_path.read_bytes())
assert len(prior["claims_rechecked"]) == 33 and all(c["status"] == "verified" for c in prior["claims_rechecked"])
assert prior["source_equivalence"]["source_sha256"] == page["source_sha256"]
result = {
    "started_at": started,
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": original["reviewer_task"],
    "lesson_id": "W.1",
    "original_report": {"path": str(R.relative_to(ROOT)), "sha256": sha(R)},
    "latest_source_freeze": {"path": str(F.relative_to(ROOT)), "sha256": sha(F)},
    "latest_canonical_source_freeze": {"path": str(C.relative_to(ROOT)), "sha256": sha(C)},
    "current_w1_snapshot": {"path": page["snapshot"], "sha256": page["source_sha256"]},
    "w1_same_exact_bytes_as_original_and_v4": True,
    "current_parent_sha256": parent["source_sha256"],
    "figure_sha256": {},
    "direct_repository_sources_unchanged": dependencies,
    "canonical_notebook_hashes_checked": len(notebook_pages),
    "all_current_notebooks_same_as_inspected_fix06_bundle": True,
    "isolated_export_latest_input_files_equal_current": True,
    "claims_recheck_basis": {"path": str(prior_path.relative_to(ROOT)), "sha256": sha(prior_path), "claim_ids": [c["claim_id"] for c in prior["claims_rechecked"]]},
    "isolated_export_latest": {"path": str(exports_path.relative_to(ROOT)), "sha256": sha(exports_path)},
    "shared_stage_stale_and_preserved": exports["stale_shared_downloads_preserved"],
    "decision": "All33 W.1 claims retain their original scope under the latest freeze; dependency drift is closed by the genuine9.11 run plus independently verified331 bundle and the latest isolated export. No shared stage update or deployment is claimed.",
    "scope_limit": "Only W.1 is signed. Other first-steps/course text was copied as export input, without substantive review or claiming fresh review of other W sections. No new reader proof is inferred.",
}
p = A / "adopt-v5-current-v1.json"
with p.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"path": str(p.relative_to(ROOT)), "sha256": sha(p), "created_at": result["created_at"], "canonical_notebook_hashes_checked": len(notebook_pages), "w1_source_sha256": page["source_sha256"]}, ensure_ascii=False))
