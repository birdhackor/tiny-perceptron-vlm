"""Use the real v2 API for group6 only; do not manufacture a completed seal.

The API object is initialized with actual files for bounded prior/row/page
checks. Its constructor/group seal path is deliberately not replaced or mocked.
An actual in-progress phase is used for the phase precondition check, which
must report pending rather than a completed historical chain.
"""

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OWNER = "/root/omni_technical_coordinator/technical_glossary"
REPORT_DIR = "docs/course-revision-20261029-omni/reviews/technical/technical_glossary"
APPENDIX = REPORT_DIR + "/technical-historical-appendix-rv1019-technical-glossary-v2.json"
FULL = REPORT_DIR + "/technical-current-v2.json"
MANIFEST = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v4.json"
COLLECTION = "docs/course-revision-20261029-omni/reviews/technical-stage/collection-intake.json"
REQUEST = "docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json"


def read(path):
    return json.loads((ROOT / path).read_bytes())


def reference(path):
    return {"path": str(path), "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


spec = importlib.util.spec_from_file_location("glossary_historical_actual_api", ROOT / "docs/review-tools/incremental_revision.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
v2 = api.module("glossary_historical_actual_v2", ROOT / "docs/review-tools/historical_fix_rechecks_v2.py")
manifest = read(MANIFEST)
request = read(REQUEST)["groups"][6]
appendix, full = read(APPENDIX), read(FULL)
assert appendix["old_owner"] == request["old_owner"] == "/root/rv1019_technical_glossary"
assert appendix["prior_generation"] == request["prior_generation"] == "accepted_20261027"
assert len(appendix["historical_issue_rechecks"]) == len(request["items"]) == 4
assert appendix["full_review"] == reference(FULL)
assert api.owner(appendix) == api.owner(full) == OWNER
freeze = api.document(ROOT, appendix["source_freeze"])
live = {p["page_id"]: p for p in freeze["pages"]}
assert appendix["source_freeze"] == manifest["source_freeze"]
assignments = api.document(ROOT, manifest["assignments"])
current_dispositions = api.document(ROOT, manifest["dispositions"])
collection_raw = (ROOT / COLLECTION).read_bytes()
collection = json.loads(collection_raw)
assert collection["role"] == "technical"
assert collection["status"] == "in_progress", "Do not silently adapt this probe to a new actual phase"

# No fake phase/manifest or monkeypatch: use actual current source, assignment,
# disposition and collection bytes; only call the named bounded API methods.
checker = v2.Rechecks.__new__(v2.Rechecks)
checker.root, checker.manifest, checker.live, checker.api = ROOT, manifest, live, api
checker.tasks = {x["task"]: x["role"] for x in assignments["tasks"]}
checker.assignments = {x["task"]: x for x in assignments["tasks"]}
checker.sealed = {collection["wave"]: {"technical": collection}}
checker.decisions = {v2._legacy.issue_key(x): x for x in current_dispositions["items"]}
checker.frozen = live
checker.entries, checker.used, checker.reports = {}, set(), []
checker.checked_readers = set()
checks = []
checker.prior(request["prior_active_pointer"])
checks.append({"method": "Rechecks.prior", "status": "passed_bounded_check",
               "meaning": "Actual prior pointer is in active history, authentic completed prior gate/freeze/dispositions/assignments checked. This is not a newly completed current gate."})
assert checker.old_tasks[request["old_owner"]]["role"] == "technical"
checker.unavailable(request["unavailability"], request["old_owner"], api.at(appendix["created_at"]))
checks.append({"method": "Rechecks.unavailable", "status": "passed_bounded_check",
               "meaning": "Exact old-task followup-not-found plus empty listing, declared recorder and true prior<=record<=appendix chronology checked."})
checker.validate_page("glossary", OWNER, full, appendix)
checks.append({"method": "Rechecks.validate_page", "status": "passed_bounded_check",
               "meaning": "Real current full report and separate appendix both pass original technical validator against exact fix13 glossary and figures; no coverage/truth inferred from schema."})
rows = []
for index, (item, row) in enumerate(zip(request["items"], appendix["historical_issue_rechecks"])):
    for key, value in item.items():
        assert row[key] == value, (index, key)
    assert row["prior_generation"] == request["prior_generation"]
    assert row["old_owner"] == request["old_owner"] and row["old_role"] == "technical"
    assert row["necessity"] == "necessary"
    checker.closed_item(request, row, full, appendix, OWNER, "glossary", api.at(appendix["created_at"]))
    disposition = api.node(checker.old_dispositions, item["disposition_ref"]["json_pointer"])
    assert checker.page(disposition, item["historical_fix_verification"], "glossary")
    evidence = row["pages"]["glossary"]
    assert evidence["source_sha256"] == live["glossary"]["source_sha256"]
    assert evidence["figures_sha256"] == live["glossary"]["figures_sha256"]
    raw = api.artifact(ROOT, {"path": live["glossary"]["snapshot"], "sha256": live["glossary"]["source_sha256"]}).decode()
    assert evidence["source_quote"] in raw
    original_dict = api.node(api.document(ROOT, row["exact_original_disposition_snapshot"]["report"]), row["exact_original_disposition_snapshot"]["json_pointer"])
    assert original_dict == disposition
    rows.append({"index": index, "issue_id": item["issue_id"], "disposition_ref": item["disposition_ref"],
                 "original_report": item["original_report"], "json_pointer": item["json_pointer"],
                 "historical_fix_verification": item["historical_fix_verification"], "necessity": "necessary",
                 "current_quote_present": True, "original_dict_equal": True, "closed_item_API": "passed_bounded_check"})
checker.finish()
checks.append({"method": "Rechecks.closed_item / page / finish", "status": "passed_bounded_check",
               "scope": "Exactly the four group6 rows and their original immutable identities only; each bounded entry used once. Not constructor/chains/group/stage validation."})

phase_result = {}
assignment = checker.assignments[OWNER]
try:
    checker.phase_receipt(reference(APPENDIX), OWNER, assignment["wave"], "technical", assignment["dispatch_window"])
except ValueError as error:
    assert str(error) == "historical recheck: collection phase incomplete", str(error)
    phase_result = {"status": "pending", "actual_API_error": str(error),
                    "actual_phase_status": collection["status"],
                    "reason": "No completed technical seal and no claimed received_at/appendix seal fabricated; current full-group validation remains pending actual assembly."}
else:
    raise AssertionError("Actual in-progress collection unexpectedly accepted as sealed")
preserved = {
    REPORT_DIR + "/technical-original-v1.json": "f41987fdfb492664e559e32d5f1b774e57e7464057d989ccb61b17480e981819",
    REPORT_DIR + "/technical-current-v1.json": "b3b45f2ccd0f2b3a8624ade5823f20c174116d080d4efb1f9743641be5a54341",
    FULL: "f84d63c0934b2db16b823e1f0f50b4c662f119d526f8437b7d9e7e9aeb8b8680",
    REPORT_DIR + "/technical-historical-appendix-rv1019-technical-glossary-v1.json": "44587b25a00725ce1ca9290ea9c489b2694c38ec610cc69b451977c1b68a4b5e",
    "docs/technical-reviews/artifacts/omni-technical-glossary-v1/numbered-intake-current-v1.json": "e1508e7b54e7117d85638cbc8b3eccc44b75cfa8242614cbdbfcb3afc7ac1a54",
    "docs/technical-reviews/artifacts/omni-technical-glossary-v1/numbered-intake-current-v2.json": "97109393f7e9a449789deb3e9a3b0df36f46f03bfcb306325764cc199e55e551",
}
assert all(reference(p)["sha256"] == sha for p, sha in preserved.items())
output = {
    "created_at": datetime.now(timezone.utc).isoformat(), "reviewer_task": OWNER,
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-glossary-v1/historical-rv1019-glossary-api-v1.py",
    "status": "bounded_group6_checks_passed_full_chain_collection_pending",
    "request": {"report": reference(REQUEST), "json_pointer": "/groups/6"},
    "appendix": reference(APPENDIX), "full_review": reference(FULL),
    "manifest_context": reference(MANIFEST), "source_freeze": appendix["source_freeze"],
    "actual_current_dispositions": manifest["dispositions"], "actual_assignments": manifest["assignments"],
    "actual_collection_observation": {"path": COLLECTION, "observed_sha256": hashlib.sha256(collection_raw).hexdigest(),
                                      "mutable_live_file": True, "status": collection["status"],
                                      "observed_existing_entries": len(collection["collection_index"]),
                                      "note": "Observed bytes fingerprint, not a permanent immutable-input ref; subsequent legitimate collection appends do not overwrite this historical observation."},
    "API_versions": [reference(p) for p in ["docs/review-tools/incremental_revision.py", "docs/review-tools/historical_fix_rechecks.py", "docs/review-tools/historical_fix_rechecks_v2.py"]],
    "bounded_API_checks": checks, "rows": rows, "bounded_entries": len(checker.entries), "bounded_used_entries": len(checker.used),
    "actual_phase_receipt_API": phase_result,
    "preserved_immutable_refs": [{"path": p, "sha256": sha} for p, sha in preserved.items()],
    "not_executed_or_claimed": ["Rechecks constructor/full group/chains validation without true new seal and current canonical collector assembly", "global history gate pass", "other groups or continuity signatures", "new reader/coherence", "GPU/training/paid run/commit"],
}
path = OUT / "historical-rv1019-glossary-api-v1.json"
with path.open("x") as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"receipt": reference(path.relative_to(ROOT)), "status": output["status"], "bounded_rows": len(rows),
                  "phase_receipt": phase_result}, ensure_ascii=False))
