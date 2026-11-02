"""Bounded offline native parsers; no gate, candidate, receipt, or seal.

The original-report fixture exists only in memory to exercise native schema.
It is not a saved reviewer judgment, assignment, or collection record.
"""

import copy
import hashlib
import importlib.util
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DIRECTORY = Path(__file__).resolve().parent
TASK = "/root/c_v3_technical_coord/current_data_auditor"
CONTRACT = "docs/course-revision-20261101-c-v3/technical-coordination-v21/CURRENT-DATA-AUDIT-CONTRACT.md"
FREEZE = "docs/course-revision-20261101-c-v3/technical-coordination-v21/current-data-contract-freeze-v1.json"
CONTRACT_SHA = "886e50c9dc52cc4299d2606de250705bd46ef27d0fec465038af2cfc69064ac1"
FREEZE_SHA = "65041ad3eec8bff86b51054264171cd181def4f49c11ee1590bd27cca9abe4bb"
EXPECTED_INPUTS = {
    CONTRACT: CONTRACT_SHA,
    FREEZE: FREEZE_SHA,
    "docs/review-tools/incremental_revision.py": "654b5796af2be03deed1407ce49d3537dc663208db27d43811896f59b624608d",
    "scripts/omni_revision/review_setup.py": "701a87eb97fe51d2449b88602c4814236bb73297dce1494964e6102ff6684ad0",
    "scripts/check_technical_reviews.py": "0f267b62a23d7dbcb630a7cffb80098e881b51566f8370f0f76197e4bff41401",
    "docs/review-tools/historical_fix_rechecks.py": "bcb9ecc664b3cdd2b8ddbfe8c0e89fc218a7c40bc6406c28bb92feac7e2d93f9",
    "docs/review-tools/historical_fix_rechecks_v2.py": "b551d9ef6ca0e5f47cd4dd7b27c7dda0361b47d4c099775b6382088c543a04a3",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    started = datetime.now(UTC).isoformat()
    inputs = {name: sha((ROOT / name).read_bytes()) for name in EXPECTED_INPUTS}
    assert inputs == EXPECTED_INPUTS, "contract source drift"
    api = module("current_data_native_api_probe", "docs/review-tools/incremental_revision.py")
    setup = module("current_data_native_setup_probe", "scripts/omni_revision/review_setup.py")
    validator = module("current_data_native_schema_probe", "scripts/check_technical_reviews.py")
    cases = []

    def record(name, detail):
        cases.append({"case": name, "result": "expected behavior observed", "detail": detail})

    def rejects(name, call, fragment):
        try:
            call()
        except (ValueError, KeyError, TypeError) as error:
            assert fragment in str(error), str(error)
            record(name, str(error))
        else:
            raise AssertionError(name + " unexpectedly accepted")

    contract_ref = {"path": CONTRACT, "sha256": CONTRACT_SHA}
    freeze_ref = {"path": FREEZE, "sha256": FREEZE_SHA}
    assert api.artifact(ROOT, contract_ref) == (ROOT / CONTRACT).read_bytes()
    assert api.document(ROOT, freeze_ref)["pages"][0]["page_id"] == "__current_data__"
    record("exact_hash_and_json", "Pinned contract bytes read; pinned supplemental freeze parsed.")
    rejects("wrong_hash", lambda: api.artifact(ROOT, {"path": CONTRACT, "sha256": "0" * 64}), "artifact changed")
    rejects("escape_path", lambda: api.local(ROOT, "../outside.json"), "escapes repository")
    rejects("absolute_path", lambda: api.local(ROOT, str(ROOT / CONTRACT)), "relative artifact path")
    assert api.node({"a/b": {"~key": [9]}}, "/a~1b/~0key/0") == 9
    record("json_pointer_escape", "Escaped keys and list index resolve to the expected value.")
    rejects("bad_json_pointer", lambda: api.node({}, "issues/0"), "invalid JSON pointer")
    assert api.at(started).tzinfo is not None
    rejects("naive_timestamp", lambda: api.at("2026-11-01T00:00:00"), "requires timezone")
    assert api.at("2099-01-01T00:00:00+00:00") > datetime.now(UTC)
    record("timestamp_parser_scope", "at() parses an aware future scalar; it is not a no-future chronology validator. No seal was created.")
    sample = {"pages": [{"issues": [{"id": "x"}, "not an issue object"]}], "meta": {"new_issues": [{"id": "y"}]}}
    assert api.reported_issues(sample) == ["/pages/0/issues/0", "/meta/new_issues/0"]
    record("issue_pointer_inventory", "Nested issues/new_issues objects yield exact pointers; a string element is ignored.")
    fixture = {
        "schema_version": 1, "review_stage": "technical", "lesson_id": "__current_data__",
        "source": CONTRACT, "source_sha256": CONTRACT_SHA, "figure_sha256": {},
        "reviewer_task": TASK, "reviewer_context": "fresh", "created_at": datetime.now(UTC).isoformat(),
        "verdict": "revise",
        "claims": [{"id": "fixture_pending", "kind": "software", "statement": "Schema-only fixture: actual C data has not been supplied.",
                    "location": "in-memory fixture only", "scope": "Not a saved actor report or collection record.",
                    "status": "unresolved", "evidence": [], "artifact_ids": []}],
        "sources": [], "artifacts": [],
        "issues": [{"claim_id": "fixture_pending", "status": "unresolved", "details": "Fixture only."}],
        "checks": {key: {"status": "revise", "details": "In-memory schema-only pending fixture.", "claim_ids": ["fixture_pending"]} for key in validator.CHECKS},
    }

    def check_original(value):
        setup.check_original(value, freeze_ref, "__current_data__", "__current_data__", TASK, validator, api)

    check_original(fixture)
    record("original_revise_is_preserved", "Native check_original accepts a complete unresolved/revise schema fixture against the administrative freeze.")
    wrong_owner = copy.deepcopy(fixture)
    wrong_owner["reviewer_task"] = TASK + "_wrong"
    rejects("original_wrong_owner", lambda: check_original(wrong_owner), "身分不完整")
    wrong_source = copy.deepcopy(fixture)
    wrong_source["source_sha256"] = "0" * 64
    rejects("original_wrong_source", lambda: check_original(wrong_source), "身分不完整")
    false_pass = copy.deepcopy(fixture)
    false_pass["verdict"] = "pass"
    rejects("original_false_pass", lambda: check_original(false_pass), "仍有未決主張／問題")
    source_subset = {"sources": [{"kind": "repository_code", "path": CONTRACT, "sha256": CONTRACT_SHA}]}
    snapshot = {"source": CONTRACT, "snapshot": contract_ref}
    assert setup.original_repository_versions(source_subset, [snapshot], api) == {CONTRACT: [CONTRACT_SHA]}
    record("registered_original_snapshot", "Saved byte identity already declared by the source subset is accepted.")
    rejects("duplicate_original_snapshot", lambda: setup.original_repository_versions(source_subset, [snapshot, snapshot], api), "唯一程式版本")
    rejects("unregistered_original_snapshot", lambda: setup.original_repository_versions(source_subset, [{"source": CONTRACT, "snapshot": freeze_ref}], api), "唯一程式版本")
    software = {"id": "schema_fixture", "kind": "software", "statement": "Fixture checks rejection of missing execution evidence.",
                "location": "in-memory schema fixture", "scope": "Not a real review claim.", "status": "verified",
                "evidence": [{"source_id": "s", "locator": "fixture", "supports": "fixture"}], "artifact_ids": [],
                "verification": {"method": "executed", "expected": "fixture", "observed": "fixture", "details": "fixture"}}
    errors = []
    validator._claims([software], {"s": {"kind": "repository_code"}}, {}, errors)
    assert any("executed需引用execution artifact" in error for error in errors)
    record("software_without_execution_artifact", "; ".join(errors))
    output = {
        "kind": "bounded_offline_native_contract_probe_not_review_or_collection", "created_by": TASK,
        "started_at": started, "completed_at": datetime.now(UTC).isoformat(),
        "status": "all expected bounded probe behaviors observed", "input_sha256s": inputs,
        "probe_code_sha256": sha(Path(__file__).read_bytes()),
        "environment": {"python": platform.python_version(), "executable": sys.executable,
                        "runner": "uv run --extra cpu --no-sync", "UV_PROJECT_ENVIRONMENT": os.environ.get("UV_PROJECT_ENVIRONMENT", ""),
                        "UV_OFFLINE": os.environ.get("UV_OFFLINE", ""), "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE", ""),
                        "device": "CPU; standard-library parser only; no torch import or training"},
        "cases": cases,
        "scope": "Only pinned native parsing/schema primitives and in-memory schema counterexamples. No audit_manifest, full gate, candidate, numbered bundle, dispatch, receipt, seal, historical data chain or final-data review was executed or manufactured.",
    }
    path = DIRECTORY / "contract-probe-results-v1.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes()), "cases": len(cases), "status": output["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
