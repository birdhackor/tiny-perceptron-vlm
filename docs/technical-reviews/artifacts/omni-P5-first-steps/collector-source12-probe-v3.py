"""Same-owner source12 bounded callback; files, AST identity and saved outputs only."""
from datetime import datetime, timezone
from pathlib import Path
import difflib
import hashlib
import json
import re

from scripts.export_course import checked_notebook
from scripts.omni_revision import review_setup as setup
from scripts.reading_time import lesson_slices

ROOT=Path(__file__).resolve().parents[4]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ref(path):
    return {"path":path,"sha256":sha(ROOT/path)}
def read_ref(r):
    p=ROOT/r["path"]
    assert sha(p)==r["sha256"],r["path"]
    return json.loads(p.read_bytes())
def at(s):
    t=datetime.fromisoformat(s)
    assert t.utcoffset() is not None
    return t

prior_ref={"path":"docs/course-revision-20261029-omni/reviews/P5-actual/technical-first-steps/technical-current-v2.json","sha256":"268edef508a83e59986101c6964bbc3c6a4cc17db11144445eb6204e8db3109f"}
prior=read_ref(prior_ref)
repair_ref={"path":"docs/course-revision-20261029-omni/reviews/P5-actual/coherence-three-author-repairs-source12-v1.json","sha256":"76e29aef2e528bf06ae20eacfdf48f28327fa9432e6477d50996be44301a3ea7"}
repairs=read_ref(repair_ref)
side_ref={"path":"docs/technical-reviews/artifacts/omni-P5-numbered-v3/numbered-section-reviews-v1.json","sha256":"5225b24a74b01fe824481d792c61d299f6400c3141c1bfd185ed196b432ec158"}
side=read_ref(side_ref)
old_side=read_ref(prior["section_sidecar_crosscheck"]["sidecar"])
canonical=read_ref(side["canonical_source_freeze"])
assert side["canonical_source_freeze"]==repairs["source_freeze"]
assert side["canonical_source_freeze"]["sha256"]=="1d95d71c09a20a07334cd30707d61bcfe7a6334f8bc12241b04ee2dc66aed4b1"
numbered=read_ref(side["section_freeze"])
assert side["section_freeze"]["sha256"]=="f7ba7cc4df7edfb693f2024f819e8886a0fabba56bdfb0b6b94f18c364551423"
assert numbered["canonical_source_freeze"]==side["canonical_source_freeze"]
old_canonical=read_ref(prior["source_freeze"])
pages={x["page_id"]:x for x in canonical["pages"]}
old_pages={x["page_id"]:x for x in old_canonical["pages"]}
changed_pages=[k for k in old_pages if old_pages[k]!=pages[k]]
assert set(changed_pages)=={"9.17","12.17","19.23"}
page=pages["first-steps"]
assert page==old_pages["first-steps"]
body=(ROOT/page["snapshot"]).read_bytes()
assert (ROOT/page["source"]).read_bytes()==body
assert hashlib.sha256(body).hexdigest()==page["source_sha256"]==prior["source_sha256"]
assert page["figures_sha256"]==prior["figure_sha256"]
for path,digest in page["figures_sha256"].items():assert sha(ROOT/path)==digest
for artifact in prior["artifacts"]:assert sha(ROOT/artifact["path"])==artifact["sha256"]
for source in prior["sources"]:
    if source["kind"]=="repository_code":assert sha(ROOT/source["path"])==source["sha256"]
old_entries={x["lesson_id"]:x for x in old_side["sections"]}
entries={x["lesson_id"]:x for x in side["sections"]}
changed_entries=[k for k in old_entries if old_entries[k]!=entries[k]]
assert changed_entries==["W.7"]
old_w7=read_ref(old_entries["W.7"]["report"])
new_w7=read_ref(entries["W.7"]["report"])
assert old_w7["reviewer_task"]==new_w7["reviewer_task"]=="/root/omni_p5_technical_w_7"
assert [(x["id"],x["kind"],x["statement"],x["status"]) for x in old_w7["claims"]]==[(x["id"],x["kind"],x["statement"],x["status"]) for x in new_w7["claims"]]
assert old_w7["issues"]==new_w7["issues"]==[]
old_sources={x["path"]:x for x in old_w7["sources"] if x["kind"]=="repository_code"}
new_sources={x["path"]:x for x in new_w7["sources"] if x["kind"]=="repository_code"}
changed_repo=[p for p in old_sources if old_sources[p]["sha256"]!=new_sources[p]["sha256"]]
assert set(changed_repo)=={"course/chapters/09.md","course/chapters/12.md","course/chapters/19.md"}
diff_rows=[]
for change in repairs["changes"]:
    sid=change["page_id"]
    before=(ROOT/change["before"]["path"]).read_text()
    after=(ROOT/change["after"]["path"]).read_text()
    assert hashlib.sha256(before.encode()).hexdigest()==change["before"]["sha256"]
    assert hashlib.sha256(after.encode()).hexdigest()==change["after"]["sha256"]
    actual_diff="".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="source11/"+sid,tofile="source12/"+sid,n=2))
    assert actual_diff==change["diff"]
    path=pages[sid]["source"]
    text=(ROOT/path).read_text()
    assert lesson_slices(text)[sid]==after
    assert text.count(after)==1
    restored=text.replace(after,before,1)
    assert hashlib.sha256(restored.encode()).hexdigest()==old_sources[path]["sha256"]
    assert sha(ROOT/path)==new_sources[path]["sha256"]
    before_code=re.findall(r"```python\n(.*?)```",before,re.S)
    after_code=re.findall(r"```python\n(.*?)```",after,re.S)
    if sid!="9.17":assert before_code==after_code
    diff_rows.append({"section":sid,"chapter":path,"actual_diff":actual_diff,"previous_chapter_sha256":old_sources[path]["sha256"],"current_chapter_sha256":new_sources[path]["sha256"],"restored_previous_chapter_exact":True,"python_blocks_identical":before_code==after_code,"scope":"Personally read the actual diff with adjacent paragraph/code; verify only its effect on first-steps source and navigation, not a full review of this changed lesson."})

slices=lesson_slices(body.decode())
old_numbered=read_ref(old_side["section_freeze"])
old_sections={x["page_id"]:x for x in old_numbered["pages"]}
new_sections={x["page_id"]:x for x in numbered["pages"]}
assert len(new_sections)==26
for sid in new_sections:
    assert new_sections[sid]["source_sha256"]==old_sections[sid]["source_sha256"]
    assert new_sections[sid]["figures_sha256"]==old_sections[sid]["figures_sha256"]
    assert (ROOT/new_sections[sid]["snapshot"]).read_bytes()==(ROOT/old_sections[sid]["snapshot"]).read_bytes()
validator=setup.module("first_steps_source12_validator_v3","scripts/check_technical_reviews.py")
api=setup.module("first_steps_source12_incremental_v3","docs/review-tools/incremental_revision.py")
old_content_proof=json.loads((ROOT/"docs/technical-reviews/artifacts/omni-P5-first-steps/collector-current-proof-v1.json").read_bytes())
owners={x["task"]:[x["lesson_id"]] for x in old_content_proof["sections"]}
section_rows=[]
for sid in slices:
    entry=entries[sid]
    report=read_ref(entry["report"])
    original=read_ref(entry["original_report"])
    assert (ROOT/new_sections[sid]["snapshot"]).read_text()==slices[sid]
    errors=validator._validate(ROOT,ROOT/page["source"],sid,slices[sid],report,owners,set())
    assert not errors,(sid,errors)
    setup.check_original(original,entry["original_section_freeze"],sid,"first-steps",report["reviewer_task"],validator,api,entry.get("original_source_snapshots",[]))
    section_rows.append({"lesson_id":sid,"source_sha256":report["source_sha256"],"current_report":entry["report"],"original_report":entry["original_report"],"validate_errors":errors,"original_check":"passed existing check_original; original preserved","substantive_basis":"Unchanged 194 claims reused from own post-original current-v1 crosscheck; W.7 source12 bounded proof and three real source diffs additionally read."})
old_inv=read_ref(old_side["issue_inventory"])
new_inv=read_ref(side["issue_inventory"])
assert old_inv["issues"]==new_inv["issues"]
removed=[x for x in old_inv["reports"] if x not in new_inv["reports"]]
added=[x for x in new_inv["reports"] if x not in old_inv["reports"]]
assert removed==[old_entries["W.7"]["report"]] and added==[entries["W.7"]["report"]]

stage_ref=ref("docs/course-revision-20261029-omni/reviews/P5-actual/coherence-stage-open-v1.json")
stage=read_ref(stage_ref)
technical_ref=stage["preceding_technical_seal"]
technical=read_ref(technical_ref)
assert technical_ref["sha256"]=="b756abd42038c1baf010b0ee64de67cd016e903f0fc03371322ec5c64b35be64"
assert technical["status"]=="completed" and technical["role"]=="technical"
assert at(technical["completed_at"])<=at(stage["started_at"])
collection_ref=ref("docs/course-revision-20261029-omni/reviews/P5-actual/coherence-current-collection-intake-v3.json")
collection=read_ref(collection_ref)
dispatch_ref=ref("docs/course-revision-20261029-omni/reviews/P5-actual/source12-callback-dispatch-batch1-v1.json")
dispatch=read_ref(dispatch_ref)
entry=entries["W.7"]
assert entry["current_dispatch_window"]==dispatch["dispatch_window"]
assert {"task":new_w7["reviewer_task"],"actual_followup_tool_acknowledged":True} in dispatch["tasks"]
matches=[x for x in collection["collection_index"] if x["path"]==entry["report"]["path"] and x["sha256"]==entry["report"]["sha256"]]
assert len(matches)==1
received=matches[0]
assert received["received_at"]==entry["current_collection"]["received_at"]
assert received["collection_role"]==entry["current_collection"]["collection_role"]=="continuity"
assert at(technical["completed_at"])<=at(stage["started_at"])<=at(dispatch["dispatch_window"]["not_before"])<=at(new_w7["created_at"])<=at(received["received_at"])<=at(side["created_at"])<=datetime.now(timezone.utc)
assert not [x for x in technical["collection_index"] if x["path"]==entry["report"]["path"]]
own_dispatch_ref=ref("docs/course-revision-20261029-omni/reviews/P5-actual/source12-callback-dispatch-batch2-open-v1.json")
own_dispatch=read_ref(own_dispatch_ref)
assert own_dispatch["collection_role"]=="continuity"

cpu_receipt=read_ref(repairs["execution"])
assert cpu_receipt["source_freeze"]==side["canonical_source_freeze"]
bundle=ROOT/"outputs/omni-P5-current-kernel-bundle-v10"
notebooks=[]
for pid,p in pages.items():
    if "notebook" not in p:continue
    source=ROOT/p["notebook"]
    assert sha(source)==p["notebook_sha256"]
    executed=bundle/Path(p["notebook"]).relative_to("notebooks")
    nb=checked_notebook(json.loads(source.read_bytes()),executed)
    notebooks.append({"id":pid,"notebook":p["notebook"],"notebook_sha256":sha(source),"preserved_output_sha256":sha(executed),"code_cells":sum(x["cell_type"]=="code" for x in nb["cells"])})
assert len(notebooks)==331

# Old map targets: 66 pre-Omni unchanged; the added 19.23 prose changed only.
for row in old_content_proof["original_optional_issue_resolution"]["old_66_map_targets"]:
    original_inspection=json.loads((ROOT/"docs/technical-reviews/artifacts/omni-P5-first-steps/inspection-output-v1.json").read_bytes())
    loc=next(x for r in original_inspection["w7_navigation_rows"] if r["module"]==row["module"] for x in r["references"] if x["section"]==row["section"])
    assert hashlib.sha256(lesson_slices((ROOT/loc["source"]).read_text())[loc["section"]].encode()).hexdigest()==loc["section_sha256"]
for row in old_content_proof["original_optional_issue_resolution"]["omni_definition_comparisons"]:
    assert sha(ROOT/row["module"])==row["module_sha256"]
print(json.dumps({"created_at":datetime.now(timezone.utc).isoformat(),"reviewer_task":prior["reviewer_task"],"callback_kind":"source12_same_original_owner_bounded_reuse_during_continuity03_not_fresh_whole_blind_reading","prior_current_report":prior_ref,"repair_provenance":repair_ref,"canonical_source_freeze":side["canonical_source_freeze"],"section_freeze":side["section_freeze"],"sidecar":side_ref,"issue_inventory":side["issue_inventory"],"assigned_page_identical":page,"canonical_changed_page_ids":changed_pages,"changed_section_entry_ids":changed_entries,"all26_section_bodies_and_figures_identical":True,"other25_entries_identical":True,"w7_current_original_owner_retained":new_w7["reviewer_task"],"three_dependency_diffs":diff_rows,"direct_own_repository_sources_and_prior16_artifacts_unchanged":True,"seven_W_checks":section_rows,"issue_inventory_reports_removed":removed,"issue_inventory_reports_added":added,"issue_inventory_issues_unchanged":True,"currentW7_continuity_receipt":{"stage":stage_ref,"completed_technical_seal":technical_ref,"dispatch":dispatch_ref,"dispatch_window":entry["current_dispatch_window"],"actual_collection":collection_ref,"unique_actual_receipt":received,"new_report_absent_from_closed_technical_seal":True},"own_callback_dispatch_started":own_dispatch_ref,"current331_preserved_output_association":{"receipt":repairs["execution"],"count":len(notebooks),"code_cells":sum(x["code_cells"] for x in notebooks),"notebooks":notebooks,"author_receipt_counts":{k:cpu_receipt[k] for k in ["current_fresh_revised_lessons","new_source12_CPU_kernels","markdown_only_overlays","earlier_exact_source_executions"]},"scope":"Read saved author outputs/cells only, no execution; source12 adds1 raw-score illustration CPU kernel and2 markdown-only overlays to existing record; not331 fresh reviewer kernels."},"substantive_reuse":"Own independent original plus current-v1 full seven-section post-original crosscheck retained. Current-v3 newly reads the three actual diffs and W.7 same-owner source12 callback/receipts and bindings. No full new reading or full technical review of9.17/12.17/19.23 claimed.","phase_scope":"technical03 completed seal remains immutable; this bounded callback is continuity03. No whole seal/collection, final check-numbered completion, NN/kernel, training, Modal, dispatch or publishing performed."},ensure_ascii=False,indent=2))
