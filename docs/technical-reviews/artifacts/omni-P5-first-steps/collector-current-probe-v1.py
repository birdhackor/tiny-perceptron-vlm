"""Read-only post-original W.1–W.7 collector checks; never runs a model."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import re

from scripts.export_course import checked_notebook
from scripts.omni_revision import review_setup as setup
from scripts.reading_time import lesson_slices

ROOT = Path(__file__).resolve().parents[4]
OWN = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-first-steps/technical-original-v1.json"
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def read_ref(ref):
    p = ROOT / ref["path"]
    assert digest(p) == ref["sha256"], ref["path"]
    return json.loads(p.read_bytes())
def ref(path):
    return {"path": path, "sha256": digest(ROOT/path)}
def at(text):
    dt = datetime.fromisoformat(text)
    assert dt.utcoffset() is not None
    return dt

original_ref = {"path": OWN.relative_to(ROOT).as_posix(), "sha256": "0d8bf7ebb7d244a1e09cb51faae76104d23bbd666851f7370465555c6df75885"}
original = read_ref(original_ref)
bundle_ref = {"path": "docs/technical-reviews/artifacts/omni-P5-numbered-v1/numbered-section-reviews-v1.json", "sha256": "d57aebb02fadc2b8b3a0f07e6dc59b1d831193dc92b4d06504b52dbda2acabd6"}
bundle = read_ref(bundle_ref)
canonical_ref = bundle["canonical_source_freeze"]
assert canonical_ref == {"path": "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v10.json", "sha256": "ddcc500db74f5df9ab3bb6fddcdc511ca177c98ce64e1a94160d1a7ccc1c35be"}
canonical = read_ref(canonical_ref)
numbered = read_ref(bundle["section_freeze"])
assert numbered["canonical_source_freeze"] == canonical_ref
page = next(x for x in canonical["pages"] if x["page_id"] == "first-steps")
current_bytes = (ROOT/page["snapshot"]).read_bytes()
assert hashlib.sha256(current_bytes).hexdigest() == page["source_sha256"]
assert current_bytes == (ROOT/page["source"]).read_bytes()
old_canonical = read_ref(original["source_freeze"])
old_page = next(x for x in old_canonical["pages"] if x["page_id"] == "first-steps")
old_bytes = (ROOT/old_page["snapshot"]).read_bytes()
current = current_bytes.decode()
old = old_bytes.decode()
added_row = '| `tiny_perceptron/omni_products/` | [19.21](chapters/19.md#19.21)的影音分支、[19.22](chapters/19.md#19.22)的聲音碼與Talker、[19.23](chapters/19.md#19.23)的新版候選判斷；完整計算都內建在各節 |\n'
assert current.count(added_row) == 1
assert current.replace(added_row, "") == old
assert page["figures_sha256"] == original["figure_sha256"]
for path, sha in page["figures_sha256"].items():
    assert digest(ROOT/path) == sha
old_slices, slices = lesson_slices(old), lesson_slices(current)
assert set(slices) == {"W."+str(i) for i in range(1, 8)}
for section in slices:
    assert slices[section] == old_slices[section] if section != "W.7" else slices[section].replace(added_row, "") == old_slices[section]
assert re.findall(r"```python\n(.*?)```", old, re.S) == re.findall(r"```python\n(.*?)```", current, re.S)
assert re.findall(r"```(?:bash|powershell)\n(.*?)```", old, re.S) == re.findall(r"```(?:bash|powershell)\n(.*?)```", current, re.S)
for source in original["sources"]:
    if source["kind"] == "repository_code":
        assert digest(ROOT/source["path"]) == source["sha256"]
for artifact in original["artifacts"]:
    assert digest(ROOT/artifact["path"]) == artifact["sha256"]

provenance_ref = {"path": "docs/course-revision-20261029-omni/reviews/P5-actual/numbered-neutral-receipt-provenance-v1.json", "sha256": "211e46f27219c436700e5743d0805a072786e2853e38e59b570656040c62c02d"}
provenance = read_ref(provenance_ref)
prior_assignments = read_ref(provenance["prior_assignment_source"])["tasks"]
dispatches = read_ref(provenance["current_dispatch_source"])["tasks"]
assignment_map = {x["lesson_id"]:x for x in provenance["numbered_task_assignments"]}
assert len(assignment_map) == 26 and len(bundle["sections"]) == 26
for assignment in assignment_map.values():
    source_rows = prior_assignments if assignment["wave"] != "omni-mainline-03-actual" else dispatches
    matches = [x for x in source_rows if x.get("task") == assignment["task"]]
    assert len(matches) == 1 and matches[0] == assignment
owners = defaultdict(list)
for assignment in assignment_map.values():
    owners[assignment["task"]].append(assignment["lesson_id"])
assert all(len(x) == 1 for x in owners.values())
assert original["reviewer_task"] not in owners
reader_tasks = {x["task"] for x in prior_assignments if x["role"] == "reader"}
reader_tasks.update(json.loads((ROOT/"docs/course-revision-20261029-omni/reviews/P5-actual/technical-inputs-packet-v4.json").read_bytes())["reader_tasks"])
api = setup.module("first_steps_current_incremental", "docs/review-tools/incremental_revision.py")
validator = setup.module("first_steps_current_validator", "scripts/check_technical_reviews.py")
current_reader = read_ref(provenance["current_wave"]["reader_completed"])
current_open = read_ref(provenance["current_wave"]["technical_open"])
current_collection = read_ref(provenance["current_wave"]["actual_collection_so_far"])
own_original_receipts = [x for x in current_collection["collection_index"] if x["path"] == original_ref["path"] and x["sha256"] == original_ref["sha256"]]
assert len(own_original_receipts) == 1
assert at(original["created_at"]) <= at(own_original_receipts[0]["received_at"])
assert at(own_original_receipts[0]["received_at"]) < at(bundle["created_at"])
setup.check_original(original, original["original_source_freeze"], "first-steps", "first-steps", original["reviewer_task"], validator, api)
assert current_reader["status"] == "completed"
assert at(current_reader["completed_at"]) <= at(current_open["started_at"])
assert provenance["current_wave"]["completed_technical_seal"] is None
assert "open" in current_open["status"]
prior_wave = next(x for x in provenance["prior_completed_waves"] if x["id"] == "omni-mainline-01")
prior_reader = read_ref(prior_wave["seals"]["reader"])
prior_technical = read_ref(prior_wave["seals"]["technical"])
assert prior_technical["status"] == "completed"
assert at(prior_reader["completed_at"]) <= at(prior_technical["started_at"])
frozen_sections = {x["page_id"]:x for x in numbered["pages"]}
section_rows = []
for entry in bundle["sections"]:
    sid = entry["lesson_id"]
    if sid not in slices:
        continue
    report, first = read_ref(entry["report"]), read_ref(entry["original_report"])
    assignment = assignment_map[sid]
    assert report["reviewer_task"] == first["reviewer_task"] == assignment["task"]
    assert assignment["role"] == "technical" and assignment["context"] == "fresh"
    assert assignment["task"] not in reader_tasks and assignment["task"] not in original["author_tasks"]
    frozen = frozen_sections[sid]
    assert (ROOT/frozen["snapshot"]).read_text() == slices[sid]
    assert hashlib.sha256(slices[sid].encode()).hexdigest() == frozen["source_sha256"] == report["source_sha256"]
    assert report["figure_sha256"] == frozen["figures_sha256"]
    errors = validator._validate(ROOT, ROOT/page["source"], sid, slices[sid], report, owners, reader_tasks)
    assert not errors, (sid, errors)
    setup.check_original(first, entry["original_section_freeze"], sid, "first-steps", first["reviewer_task"], validator, api, entry.get("original_source_snapshots", []))
    receipts = []
    for key, which in [("original_report", "original_collection"), ("report", "current_collection")]:
        rr = entry[which]
        actual = first if key == "original_report" else report
        collection = prior_technical if rr["wave"] == "omni-mainline-01" else current_collection
        matches = [x for x in collection["collection_index"] if x["path"] == entry[key]["path"] and x["sha256"] == entry[key]["sha256"]]
        assert len(matches) == 1 and matches[0]["received_at"] == rr["received_at"]
        assert rr["collection_role"] == "technical" and rr["wave"] == assignment["wave"]
        created, received = at(actual["created_at"]), at(rr["received_at"])
        assert at(assignment["dispatch_window"]["not_before"]) <= created <= received <= datetime.now(timezone.utc)
        if rr["wave"] == "omni-mainline-01":
            setup.check_receipt(api, {rr["wave"]:{"technical":prior_technical}}, entry[key], actual, rr, assignment["dispatch_window"])
            state = "completed_prior_phase_unique_receipt"
        else:
            assert at(current_open["started_at"]) <= created
            assert at(current_reader["completed_at"]) <= created
            assert received <= at(current_collection["created_at"])
            state = "actual_current_open_phase_unique_receipt_no_completion_claim"
        receipts.append({"role":key,"report":entry[key],"created_at":actual["created_at"],"received_at":rr["received_at"],"wave":rr["wave"],"check":state})
    assert at(first["created_at"]) <= at(report["created_at"])
    assert at(entry["original_collection"]["received_at"]) <= at(entry["current_collection"]["received_at"])
    section_rows.append({"lesson_id":sid,"task":report["reviewer_task"],"current_report":entry["report"],"original_report":entry["original_report"],"original_section_freeze":entry["original_section_freeze"],"source_sha256":report["source_sha256"],"figures_sha256":report["figure_sha256"],"claims":len(report["claims"]),"sources":len(report["sources"]),"claim_ids":[x["id"] for x in report["claims"]],"current_validate_errors":errors,"original_schema_sources_artifacts_claims_checks": "passed existing check_original", "current_and_original_assigned_body_identical":report["source_sha256"]==first["source_sha256"],"evidence_kind":"new_P5_task_original_or_same_owner_current" if assignment["wave"] == "omni-mainline-03-actual" else "exact_unchanged_prior_task_evidence_not_new_reading","receipts":receipts})
assert len(section_rows) == 7
inventory = read_ref(bundle["issue_inventory"])
assert bundle["issue_inventory"]["sha256"] == "53a170ec26339686cb110f593da0787d360f53af33d16ea35fb2421bba0cdb82"
assert inventory["canonical_source_freeze"] == canonical_ref
assert not [x for x in inventory["issues"] if x.get("lesson_id", "").startswith("W.")]
for entry in bundle["sections"]:
    if entry["lesson_id"] in slices:
        assert entry["report"] in inventory["reports"]
        assert entry["original_report"] in inventory["reports"]

# All Notebook operations below only read preserved author execution records.
cpu_receipt = read_ref(provenance["current_CPU_receipt"])
execution_root = ROOT/provenance["current_exact_CPU_bundle"]
notebooks = []
for p in canonical["pages"]:
    if "notebook" not in p:
        continue
    nb_path = ROOT/p["notebook"]
    assert digest(nb_path) == p["notebook_sha256"]
    executed = execution_root/Path(p["notebook"]).relative_to("notebooks")
    nb = checked_notebook(json.loads(nb_path.read_bytes()), executed)
    notebooks.append({"id":p["page_id"],"notebook":p["notebook"],"source_sha256":digest(nb_path),"preserved_execution_sha256":digest(executed),"code_cells":sum(c["cell_type"]=="code" for c in nb["cells"])})
assert len(notebooks) == 331

# Independently re-check the one added map row structurally, never execute it.
chapter19 = ROOT/"course/chapters/19.md"
chapter_slices = lesson_slices(chapter19.read_text())
targets = [
    ("19.21", "tiny_perceptron/omni_products/av.py", ["AVConfig","mel_bank","local_audio","TimeAttention","AVModel"]),
    ("19.22", "tiny_perceptron/omni_products/sound.py", ["CodecConfig","SoundCodec","program_schedule","ProgramTalker"]),
    ("19.23", "tiny_perceptron/omni_products/choice.py", ["text_bytes","Candidate","Question","pad_bytes","collate_questions","ChoiceConfig","StateMemory","ByteEncoder","ChoiceModel"]),
]
omni_rows = []
for sid, path, names in targets:
    target_page = next(x for x in canonical["pages"] if x["page_id"] == sid)
    section = (ROOT/target_page["snapshot"]).read_text()
    assert section == chapter_slices[sid]
    parsed = []
    for code in re.findall(r"```python\n(.*?)```", section, re.S):
        parsed.extend(ast.parse(code).body)
    actual = ast.parse((ROOT/path).read_text())
    comparisons = []
    for name in names:
        source_node = next(x for x in actual.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name == name)
        matches = [x for x in parsed if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name == name]
        assert matches, (sid,name)
        exclusions = {"AVModel":{"config_sha256"},"SoundCodec":{"reset_unused_training_codes"}}.get(name,set())
        expected = deepcopy(source_node)
        if isinstance(expected,ast.ClassDef):
            expected.body=[x for x in expected.body if not isinstance(x,ast.FunctionDef) or x.name not in exclusions]
        matching = [x for x in matches if ast.dump(x,include_attributes=False)==ast.dump(expected,include_attributes=False)]
        assert matching, (sid,name,"AST mismatch")
        comparisons.append({"name":name,"exact_AST_match_with_documented_noncalculation_exclusions":True,"excluded_maintenance_methods":sorted(exclusions)})
    omni_rows.append({"section":sid,"section_sha256":hashlib.sha256(section.encode()).hexdigest(),"module":path,"module_sha256":digest(ROOT/path),"definitions":comparisons})

old_map = json.loads((ROOT/"docs/technical-reviews/artifacts/omni-P5-first-steps/inspection-output-v1.json").read_bytes())["w7_navigation_rows"]
retained_map_checks=[]
for row in old_map:
    for loc in row["references"]:
        chapter = ROOT/loc["source"]
        now = lesson_slices(chapter.read_text())[loc["section"]]
        retained_map_checks.append({"module":row["module"],"section":loc["section"],"exact_section_unchanged":hashlib.sha256(now.encode()).hexdigest()==loc["section_sha256"]})
assert len(retained_map_checks)==66 and all(x["exact_section_unchanged"] for x in retained_map_checks)

print(json.dumps({"created_at":datetime.now(timezone.utc).isoformat(),"reviewer_task":original["reviewer_task"],"original_report":original_ref,"own_original_precedes_sidecar_and_received":own_original_receipts[0],"own_original_schema_with_revise_preserved":"passed existing check_original","canonical_source_freeze":canonical_ref,"numbered_sidecar":bundle_ref,"issue_inventory":bundle["issue_inventory"],"provenance":provenance_ref,"first_steps_source":{"source_sha256":page["source_sha256"],"snapshot":page["snapshot"],"only_change_from_original":added_row,"all_eight_python_fences_and_all_command_fences_identical":True,"all_original_repository_sources_and_ten_artifacts_unchanged":True,"figures_identical":page["figures_sha256"]},"sections":section_rows,"current_claims_crosschecked":sum(x["claims"] for x in section_rows),"section_issue_inventory_for_W":[],"original_optional_issue_resolution":{"json_pointer":"/issues/0","original_issue_id":"first-steps-optional-omni-module-map","status":"resolved","current_source_quote":added_row.strip(),"scope":"The original optional map omission is fixed. 17 map rows /69 expanded section references; 66 previous source sections unchanged, plus3 new structurally checked sections. This is not a technical re-review of those3 model lessons.","omni_definition_comparisons":omni_rows,"old_66_map_targets":retained_map_checks},"current_notebook_output_association":{"receipt":provenance["current_CPU_receipt"],"notebooks":notebooks,"count":len(notebooks),"code_cells":sum(x["code_cells"] for x in notebooks),"all_exact_source_and_nonerror_recorded_execution":True,"scope":"Preserved author outputs only:18 current revised kernels and313 exact earlier executions, not331 fresh reviewer executions. No model forward/backward invoked."},"phase_scope":{"current_technical_phase":"open_unsealed","completed_check_numbered_result_claimed":False,"existing_prior_phase_receipts_checked":True,"current_actual_collection_checked":True,"no_collection_or_seal_created":True},"limits":"Original whole-page independent review retained; this is bounded post-original source/map/sidecar crosscheck. W.3–W.6 are genuine prior evidence, not new P5 readings. No new source download, warmup/kernel/model execution, training, Modal, workflows, publication, or UI screenshot in this callback."},ensure_ascii=False,indent=2))
