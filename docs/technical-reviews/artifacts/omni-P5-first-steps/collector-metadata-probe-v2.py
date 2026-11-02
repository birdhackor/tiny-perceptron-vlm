"""Bounded source11/sidecar-v2 metadata callback. No model or full reread."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

from scripts.omni_revision import review_setup as setup
from scripts.reading_time import lesson_slices

ROOT = Path(__file__).resolve().parents[4]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def read_ref(ref):
    p = ROOT/ref["path"]
    assert sha(p) == ref["sha256"], ref["path"]
    return json.loads(p.read_bytes())

prior_ref = {"path":"docs/course-revision-20261029-omni/reviews/P5-actual/technical-first-steps/technical-current-v1.json", "sha256":"23b4de293d7f0fee217a67796e56d711b574b5b3f7eaf2c55d18ca82f6407082"}
prior = read_ref(prior_ref)
old_side = read_ref(prior["section_sidecar_crosscheck"]["sidecar"])
new_side_ref = {"path":"docs/technical-reviews/artifacts/omni-P5-numbered-v2/numbered-section-reviews-v1.json", "sha256":"a6b2189776362c7bc883c0ea26d01c2223b191b5989799dfa411fc48ba0dd87f"}
new_side = read_ref(new_side_ref)
assert old_side["sections"] == new_side["sections"]
old_freeze = read_ref(prior["source_freeze"])
new_freeze = read_ref(new_side["canonical_source_freeze"])
assert new_side["canonical_source_freeze"]["path"].endswith("source-freeze-v11.json")
old_pages = {x["page_id"]:x for x in old_freeze["pages"]}
new_pages = {x["page_id"]:x for x in new_freeze["pages"]}
assert old_pages.keys() == new_pages.keys()
changed_pages = [k for k in old_pages if old_pages[k] != new_pages[k]]
assert changed_pages == ["curriculum"]
assert old_pages["first-steps"] == new_pages["first-steps"]
page = new_pages["first-steps"]
body = (ROOT/page["snapshot"]).read_bytes()
assert sha(ROOT/page["snapshot"]) == page["source_sha256"] == prior["source_sha256"]
assert (ROOT/page["source"]).read_bytes() == body
assert page["figures_sha256"] == prior["figure_sha256"]
for path,digest in page["figures_sha256"].items():
    assert sha(ROOT/path) == digest
old_numbered = read_ref(old_side["section_freeze"])
new_numbered = read_ref(new_side["section_freeze"])
assert new_side["section_freeze"]["sha256"] == "1ce6261143dc8321cf9ec7bc9e298c678aab07eed5daaaf66cb8ca04dd16a68a"
assert new_numbered["canonical_source_freeze"] == new_side["canonical_source_freeze"]
slices = lesson_slices(body.decode())
old_sections = {x["page_id"]:x for x in old_numbered["pages"]}
new_sections = {x["page_id"]:x for x in new_numbered["pages"]}
assert set(slices) == {"W."+str(i) for i in range(1,8)}
for sid in slices:
    a,b=old_sections[sid],new_sections[sid]
    assert {k:v for k,v in a.items() if k != "snapshot"} == {k:v for k,v in b.items() if k != "snapshot"}
    assert (ROOT/a["snapshot"]).read_bytes() == (ROOT/b["snapshot"]).read_bytes() == slices[sid].encode()
old_inventory = read_ref(old_side["issue_inventory"])
new_inventory = read_ref(new_side["issue_inventory"])
assert old_inventory["reports"] == new_inventory["reports"]
assert old_inventory["issues"] == new_inventory["issues"]
assert new_inventory["canonical_source_freeze"] == new_side["canonical_source_freeze"]
assert new_inventory["section_freeze"] == new_side["section_freeze"]
for artifact in prior["artifacts"]:
    assert sha(ROOT/artifact["path"]) == artifact["sha256"]
repo_sources=[]
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        assert sha(ROOT/source["path"]) == source["sha256"]
        repo_sources.append({"path":source["path"],"sha256":source["sha256"]})
proof = json.loads((ROOT/"docs/technical-reviews/artifacts/omni-P5-first-steps/collector-current-proof-v1.json").read_bytes())
for row in proof["current_notebook_output_association"]["notebooks"]:
    assert new_pages[row["id"]]["notebook_sha256"] == old_pages[row["id"]]["notebook_sha256"] == row["source_sha256"]
    assert sha(ROOT/row["notebook"]) == row["source_sha256"]
for row in proof["original_optional_issue_resolution"]["omni_definition_comparisons"]:
    assert sha(ROOT/row["module"]) == row["module_sha256"]
    assert new_pages[row["section"]]["source_sha256"] == old_pages[row["section"]]["source_sha256"] == row["section_sha256"]
validator=setup.module("first_steps_metadata_validator_v2","scripts/check_technical_reviews.py")
owners={x["task"]:[x["lesson_id"]] for x in proof["sections"]}
section_rows=[]
for entry in new_side["sections"]:
    if entry["lesson_id"] not in slices:
        continue
    report=read_ref(entry["report"])
    read_ref(entry["original_report"])
    sid=entry["lesson_id"]
    errors=validator._validate(ROOT,ROOT/page["source"],sid,slices[sid],report,owners,set())
    assert not errors,(sid,errors)
    section_rows.append({"lesson_id":sid,"source_sha256":report["source_sha256"],"current_report":entry["report"],"original_report":entry["original_report"],"validate_errors":errors,"basis":"Exact unchanged report/source/artifact metadata; substantive reading and receipt checks reused from prior collector-current-proof-v1."})
print(json.dumps({"created_at":datetime.now(timezone.utc).isoformat(),"reviewer_task":prior["reviewer_task"],"callback_kind":"bounded_metadata_only_same_original_owner_not_new_blind_or_full_reading","prior_current_report":prior_ref,"canonical_source_freeze":new_side["canonical_source_freeze"],"section_freeze":new_side["section_freeze"],"sidecar":new_side_ref,"issue_inventory":new_side["issue_inventory"],"canonical_changed_page_ids":changed_pages,"assigned_page_unchanged":page,"all26_section_entries_identical":True,"all_shared_issue_report_refs_and_issues_identical":True,"seven_W_slices_exact_unchanged":section_rows,"necessary_repository_sources_unchanged":repo_sources,"all_prior14_artifacts_unchanged":True,"current331_notebook_SHA_unchanged":True,"three_omni_map_target_sections_and_modules_unchanged":True,"reused_substantive_evidence":"Original whole first-steps independent review and current-v1 bounded map/194claims/receipt/331 preserved-output checks remain unchanged. Only refs/bytes/source dependencies/validators checked in this callback; no re-reading curriculum or claiming its correction verified.","phase_scope":"technical03 remains OPEN; no completed check-numbered gate, collection, seal, publication or model/kernel execution performed."},ensure_ascii=False,indent=2))
