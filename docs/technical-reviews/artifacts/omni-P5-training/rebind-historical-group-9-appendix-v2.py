"""Bounded appendix binding after actual current2 receipt; no new semantic review or NN."""
import copy
import datetime
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).parent
D = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-training"


def ref(path):
    path = path if isinstance(path, Path) else ROOT / path
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(reference):
    raw = (ROOT / reference["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == reference["sha256"], reference["path"]
    return json.loads(raw)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


api = module("training_appendix2_api", "docs/review-tools/incremental_revision.py")
historical = module("training_appendix2_contract", "docs/review-tools/historical_fix_rechecks.py")
old_ref = ref(D / "historical-group-9-appendix-v1.json")
assert old_ref["sha256"] == "e53d8a214826307a97282e21477ea7dc979445234a227c5d922eb21d8b036855"
old = read(old_ref)
full_ref = ref(D / "technical-current-v2.json")
assert full_ref["sha256"] == "7284ce2234782759ddbea67d9bfd19522bba91cfbfa51e8951c06375592cda9c"
full = read(full_ref)
previous_full = read(old["full_review"])
intake_ref = ref("docs/course-revision-20261029-omni/reviews/P5-actual/technical-original-collection-intake-v16.json")
assert intake_ref["sha256"] == "71fd6df552b561280d3bec44215ee61454c19d36bf00d393defb48db074764e9"
intake = read(intake_ref)
matches = [(index, row) for index, row in enumerate(intake["collection_index"]) if row["path"] == full_ref["path"] and row["sha256"] == full_ref["sha256"]]
assert len(matches) == 1
index, receipt = matches[0]
assert index == 71
assert receipt["reviewer_task"] == full["reviewer_task"] == old["reviewer_task"]
assert receipt["created_at"] == full["created_at"]
assert receipt["collection_role"] == "technical" and receipt["wave"] == "omni-mainline-03-actual"
assert receipt["received_at"] == "2026-10-30T12:50:28.845671+00:00"
assert api.at(full["created_at"]) <= api.at(receipt["received_at"]) <= api.at(intake["created_at"])
actual_receipt = {"report": intake_ref, "json_pointer": "/collection_index/" + str(index), "entry": receipt}
assert full["claims"] == previous_full["claims"]
assert full["source_sha256"] == previous_full["source_sha256"] == old["source_sha256"]
assert full["figure_sha256"] == previous_full["figure_sha256"] == old["figure_sha256"] == {}
assert not full.get("historical_issue_rechecks")
freeze = read(full["source_freeze"])
page = next(row for row in freeze["pages"] if row["page_id"] == "training")
body = (ROOT / page["snapshot"]).read_bytes()
assert hashlib.sha256(body).hexdigest() == full["source_sha256"]
assert body == (ROOT / "course/training.md").read_bytes()
assert all(row["current_source_quote"] == body.decode().splitlines()[8] for row in old["historical_issue_rechecks"])
side_ref = full["numbered_section_reviews"]
assert side_ref["sha256"] == "a6b2189776362c7bc883c0ea26d01c2223b191b5989799dfa411fc48ba0dd87f"
side = read(side_ref)
assert side["canonical_source_freeze"] == full["source_freeze"]
assert side["sections"] == read(previous_full["numbered_section_reviews"])["sections"]
unchanged = []
for kind, rows in [("source", old["sources"]), ("artifact", old["artifacts"])]:
    for row in rows:
        if "path" in row:
            assert ref(row["path"])["sha256"] == row["sha256"]
            unchanged.append({"kind": kind, "id": row["id"], "path": row["path"], "sha256": row["sha256"]})
observed_at = datetime.datetime.now(datetime.UTC).isoformat()
assert api.at(receipt["received_at"]) < api.at(observed_at)
binding = {
    "created_at": observed_at, "reviewer_task": full["reviewer_task"], "kind": "Bounded metadata/actual-receipt binding of previously reviewed historical group9; no new blind or semantic review",
    "previous_appendix": old_ref, "previous_full_review": old["full_review"], "full_review": full_ref, "full_review_actual_receipt": actual_receipt,
    "source_freeze": full["source_freeze"], "shared_sidecar": side_ref, "source_sha256": full["source_sha256"], "figure_sha256": {},
    "unchanged_body_bytes": len(body), "full_current1_current2_substantive_claims_identical": True, "sidecar_section_records_identical": True, "prior_appendix_registered_evidence_unchanged": unchanged,
    "historical_semantic_basis": {"report": old_ref, "json_pointer": "/historical_issue_rechecks", "instance_count": 2, "necessity_for_both": "necessary", "verdict_for_both": "pass", "status_for_both": "still_fixed", "duplicate_relationship_preserved": True, "original_judgments_and_references_preserved": True, "new_semantic_or_initial_discovery_claimed": False},
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "execution": "JSON/hash/freeze/receipt binding only; no torch import,NN,training,network,workflow or seal"},
    "scope_limits": ["Historical issue semantics,original dispositions,duplicates,necessity and unavailability were personally read in appendix1 and are explicitly reused,not newly blind-reviewed.", "Current2 was genuinely received before this binding probe; receipt is read exactly from intake16 /collection_index/71,not guessed from a message.", "No new neural model construction/load/inference/training or Windows/GPU/MPS/Colab execution; native Windows T.11 remains pending book PR CI.", "technical03 remains open; appendix2 actual receipt and phase/group/final historical gate remain root's later work. No collection or seal created."]
}
binding_path = A / "historical-group-9-binding-v2.json"
with binding_path.open("x", encoding="utf-8") as stream:
    json.dump(binding, stream, ensure_ascii=False, indent=2)
    stream.write("\n")

report = copy.deepcopy(old)
report["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()
report["source_freeze"] = full["source_freeze"]
report["full_review"] = full_ref
report["full_review_actual_receipt"] = actual_receipt
report["numbered_section_reviews"] = side_ref
report["previous_appendix"] = old_ref
report["review_kind"] = "Same-owner bounded historical appendix2 binding after actual current2 receipt; original appendix1 semantics reused; no new blind reading"
report["artifacts"] += [
    {"id": "a_binding_code", "kind": "code", **ref(Path(__file__)), "description": "Same reviewer bounded source/freeze/sidecar/current2 receipt rebinding and existing canonical/historical contract selfcheck. Does not execute or alter original evidence or produce collection/seals."},
    {"id": "a_binding", "kind": "execution", **ref(binding_path), "description": "Actual metadata callback: exact intake16 current2 entry71,body and claim identity,shared sidecar and prior appendix evidence hashes; both prior semantic issue rows retained unchanged.", "command": "PYTHONDONTWRITEBYTECODE=1 python docs/technical-reviews/artifacts/omni-P5-training/rebind-historical-group-9-appendix-v2.py", "result": "Current2 true receipt2026-10-30T12:50:28.845671+00:00 precedes binding; source/freeze/sidecar and all prior registered evidence match; two prior necessary/still_fixed semantic rows reused without changes or NN.", "environment": binding["environment"]},
    copy.deepcopy(next(item for item in full["artifacts"] if item["id"] == "a_numbered_shared"))
]
report["sources"].append({"id": "s_binding", "kind": "execution", "title": "Bounded historical appendix2 actual-receipt binding", "verified": True, "artifact_id": "a_binding"})
next(item for item in report["sources"] if item["id"] == "s_current_body")["version"] = "Canonical source-freeze-v11; exact same body as appendix1 source-freeze-v10; metadata binding updated after actual current2 receipt"
report["checks"]["factual_accuracy"]["details"] = "本次不新增語義審閱：兩份原 necessary/fixed 實例的判斷、原處置、duplicate/unavailability與四項軟體claims沿用本人appendix1。當前body及fullcurrent1/current2之118claims一致；本次只核source11/sharedsidecar/實收current2綁定，兩個historical rows完整原樣保留。"
report["checks"]["source_verification"]["details"] = "實讀 intake16 exact SHA及/collection_index/71：current2 SHA7284ce...、created12:50:17.378246、received12:50:28.845671+00:00；再建立本appendix2。核source11與new共享sidecar exact path/SHA、training正文/引句、本人appendix1全部登記來源/artifact hashes無drift。舊issue/處置/closure語義與必要性沿用appendix1，沒有新盲讀。"
report["checks"]["limitations"]["details"] = "同owner的bounded metadata binding；兩個歷史語義判斷和既有執行沿用appendix1，不將先前parser/mock/靜態loader或歷史量化說成本次重跑。本次只有JSON/hash/receipt及契約檢查，無NN/訓練/硬體/Windows/網路/Modal/workflow/發布。Windows T.11仍待book PR CI；technical03仍OPEN，appendix2尚待root真收件，沒有phase seal或final gate宣稱。"
report["chronology"] = {
    "independent_original_created_at": old["chronology"]["independent_original_created_at"],
    "first_full_current1_actually_received_at": old["chronology"]["full_current_actually_received_at"],
    "initial_historical_semantic_probe_completed_at": old["chronology"]["historical_probe_completed_at"],
    "previous_appendix1_created_at": old["created_at"], "full_current_created_at": full["created_at"], "full_current_actually_received_at": receipt["received_at"],
    "bounded_binding_probe_completed_at": observed_at, "appendix_created_at": report["created_at"],
    "scope": "Historical semantic work occurred after current1 actual receipt; current2 was truly received before this new bounded binding and appendix2 creation. Appendix2 actual receipt is not yet claimed; no collection or seal invented."
}
report["independence"] = {
    "own_original_completed_before_other_review_or_disposition_contents": True,
    "historical_issue_semantics_first_read_after_full_current1_actual_receipt": True,
    "current2_actually_received_before_appendix2_binding": True,
    "appendix2_is_bounded_prompted_metadata_binding_not_initial_discovery_or_fresh_blind_review": True,
    "two_historical_semantic_rows_and_claims_reused_exactly_from_appendix1": True,
    "all_originals_previous_currents_and_appendix1_unchanged": True,
    "reason": "Current2 metadata callback preserves all118 independent full-page claims and the same training body. Appendix1 personally established both necessary/still_fixed issue instances; this version changes only freeze/sidecar/full-review/actual-receipt bindings with bounded supporting metadata and existing-contract validation."
}
report["bounded_binding"] = {"artifact_id": "a_binding", "previous_semantic_appendix": old_ref, "semantic_rows_json_pointer": "/historical_issue_rechecks", "fresh_blind_or_new_semantic_review": False, "shared_sidecar_referenced_in_place": side_ref}
report["unverified_scope"] = binding["scope_limits"]
report["status"] = "saved bounded appendix2; awaiting actual root collection; technical03 remains open"
assert report["historical_issue_rechecks"] == old["historical_issue_rechecks"] and report["claims"] == old["claims"]
assert api.at(receipt["received_at"]) < api.at(report["created_at"])
target = D / "historical-group-9-appendix-v2.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write("\n")

packet = json.loads((ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-inputs-packet-v4.json").read_bytes())
command = [sys.executable, "docs/course-review-20261019/tools/check_technical_report.py", "training", target.relative_to(ROOT).as_posix(), "--freeze", report["source_freeze"]["path"]]
for task in packet["reader_tasks"]:
    command += ["--reader-task", task]
completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
assert completed.returncode == 0, (completed.stdout, completed.stderr)
checker = historical.Rechecks.__new__(historical.Rechecks)
checker.root, checker.api, checker.manifest = ROOT, api, {"author_tasks": full["author_tasks"]}
checker.entries, checker.used, checker.reports, checker.checked_readers = {}, set(), [], set()
checker.prior(report["prior_active_pointer"])
checker.live, checker.frozen = {"training": page}, {"training": page}
checker.tasks = {task: "reader" for task in packet["reader_tasks"]}
checker.tasks[full["reviewer_task"]] = "technical"
checker.decisions = {historical.issue_key(item): item for item in checker.old_dispositions["items"]}
checker.validate_page("training", full["reviewer_task"], full, report)
request = read(report["request"]["report"])
requested_group = api.node(request, report["request"]["json_pointer"])
group = {"kind": "page", "page_id": "training", "unavailability": requested_group["unavailability"]}
for row in report["historical_issue_rechecks"]:
    checker.closed_item(group, row, full, report, full["reviewer_task"], "training", api.at(report["created_at"]))
assert len(checker.entries) == 2
assert ref(D / "historical-group-9-appendix-v1.json") == old_ref
assert ref(D / "technical-current-v2.json") == full_ref
assert ref(D / "technical-current-v1.json") == old["full_review"]
assert ref(D / "technical-original-v1.json") == old["independent_original_full_review"]
selfcheck = {
    "created_at": datetime.datetime.now(datetime.UTC).isoformat(), "reviewer_task": full["reviewer_task"], "report": ref(target),
    "canonical_existing_selfcheck": {"command": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
    "focused_historical_contract": {"reference": ref("docs/review-tools/historical_fix_rechecks.py"), "methods_executed": ["Rechecks.prior", "Rechecks.validate_page", "Rechecks.closed_item (calls unavailable)"], "entries_validated": 2, "errors": [], "scope": "Real prior identities/necessary preservation,current full+appendix schema,original unavailability and true current2 receipt-before-appendix chronology. closed_item decisions uses immutable accepted-prior decisions; future current-assembly disposition preservation and completed-phase collection remain untested."},
    "full_review": full_ref, "full_review_actual_receipt": actual_receipt, "previous_appendix": old_ref, "historical_semantic_rows_unchanged": True, "historical_claims_unchanged": True,
    "not_run": ["New semantic/blind review or NN/model/training/hardware/Windows/Colab/network/paid/workflow/publication.", "Rechecks group/collection/finish or final historical/check-numbered/current technical phase gate; technical03 still open and appendix2 not yet actually received."], "collection_or_seal_created": False
}
selfcheck_path = A / "historical-group-9-selfcheck-v2.json"
with selfcheck_path.open("x", encoding="utf-8") as stream:
    json.dump(selfcheck, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"appendix": {**ref(target), "bytes": target.stat().st_size, "created_at": report["created_at"], "verdict": report["verdict"]}, "full_current2_actual_receipt_pointer": actual_receipt["json_pointer"], "full_current2_actual_received_at": receipt["received_at"], "binding": {**ref(binding_path), "bytes": binding_path.stat().st_size, "created_at": observed_at}, "selfcheck": {**ref(selfcheck_path), "bytes": selfcheck_path.stat().st_size, "created_at": selfcheck["created_at"], "canonical_exit_code": completed.returncode, "historical_rows": len(checker.entries), "errors": []}}, ensure_ascii=False))
