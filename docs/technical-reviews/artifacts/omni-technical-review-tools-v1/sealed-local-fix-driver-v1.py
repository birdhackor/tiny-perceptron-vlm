"""Original owner's limited two-issue callback against a REAL completed collection.

No ledger/source/checker/report mutation. Negative cases use only deep copies.
This is not an expanded native tool original or a global gate audit.
"""

import ast
import copy
import hashlib
import importlib.util
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TASK = "/root/omni_technical_coordinator/technical_review_tools"
DIRECTORY = "docs/course-revision-20261029-omni/reviews/assembly-technical-intake-v4"
HELPER = "701a87eb97fe51d2449b88602c4814236bb73297dce1494964e6102ff6684ad0"
SEAL = "17d4f6885ecc54800e47df27f16e9d87c1a616004ca43ebec63f4175ecba7b1c"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def attempt(label, callback, expected=True):
    try:
        value = callback()
        return {"case": label, "accepted": True, "expected_accept": expected,
                "matches_expectation": expected, "result": value}
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        return {"case": label, "accepted": False, "expected_accept": expected,
                "matches_expectation": not expected, "error": str(error)}


def main():
    helper_path = ROOT / "scripts/omni_revision/review_setup.py"
    assert sha(helper_path.read_bytes()) == HELPER
    # Fresh execution of retained 61-case configuration under CURRENT imports;
    # do not relabel an old result as a new execution.
    retained_path = OUT / "local-fix-code-driver-v1.py"
    retained = load("sealed_same_owner_retained_code", retained_path)
    parsed = ast.parse(retained_path.read_text())
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    end = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "target" for target in node.targets))
    exec(compile(ast.Module(body=function.body[:end], type_ignores=[]), str(retained_path) + "#before-output", "exec"), retained.__dict__)
    rerun_code = retained.data
    assert not rerun_code["summary"]["unexpected"]
    helper = load("sealed_actual_helper", helper_path)
    api = load("sealed_actual_api", ROOT / "docs/review-tools/incremental_revision.py")
    validator = load("sealed_actual_validator", ROOT / "scripts/check_technical_reviews.py")
    base = ROOT / DIRECTORY
    manifest_path = base / "manifest-intake.json"
    packet_path = base / "packet.json"
    seal_path = base / "technical-collection-seal.json"
    assert sha(seal_path.read_bytes()) == SEAL
    assert sha(manifest_path.read_bytes()) == "b14670d80cd4490aeb10b6a564bb3f35640733b96e1391659aac71dd412f95ad"
    assert sha(packet_path.read_bytes()) == "43481fd756d1ea409166ea39a0ae379175e106960508e1e601ac4ce7daacd747"
    manifest = json.loads(manifest_path.read_bytes())
    seal = json.loads(seal_path.read_bytes())
    assert seal["status"] == "completed" and seal["role"] == "technical"
    assert api.at(seal["started_at"]) <= api.at(seal["completed_at"]) <= datetime.now(UTC)
    row_keys = [(row["path"], row["sha256"]) for row in seal["collection_index"]]
    assert len(row_keys) == len(set(row_keys)) == len(seal["report_sha256s"]) == 242
    assert set(seal["report_sha256s"]) == {row["sha256"] for row in seal["collection_index"]}
    for key in ("source_freeze", "assignments", "numbered_section_reviews"):
        api.artifact(ROOT, manifest[key])
    bundle = helper.read_ref(manifest["numbered_section_reviews"])
    index = helper.read_ref(bundle["issue_inventory"])
    section_freeze = helper.read_ref(bundle["section_freeze"])
    canonical = helper.read_ref(manifest["source_freeze"])
    tasks = helper.read_ref(manifest["assignments"])["tasks"]
    assignments = {row["task"]: row for row in tasks}
    assert len(assignments) == len(tasks)
    readers = {row["task"] for row in tasks if row["role"] == "reader"}
    _, owners = api.technical_reports(manifest, ROOT)
    section_reports = {entry["lesson_id"]: helper.read_ref(entry["report"]) for entry in bundle["sections"]}
    for sid, report in section_reports.items():
        owners.setdefault(report["reviewer_task"], []).append(sid)
    sealed = api.review_seals(ROOT, manifest["review_waves"], ("reader", "technical"))
    assert sealed["omni-mainline-01"]["technical"] == seal
    catalog = {page["page_id"]: page for page in manifest["pages"]}
    section_catalog = {page["page_id"]: page for page in section_freeze["pages"]}
    canonical_catalog = {page["page_id"]: page for page in canonical["pages"]}
    expected = helper.numbered_scope(helper.read_ref(manifest["numbered_section_requirements"]["comparison_source_freeze"]), canonical)
    assert set(expected) == {"course", "first-steps", "training", "glossary"}
    assert sum(map(len, expected.values())) == len(bundle["sections"]) == 26
    pairs = []
    for entry in bundle["sections"]:
        sid = entry["lesson_id"]
        page = section_catalog[sid]
        pairs.append((sid, page["canonical_page_id"], entry, page, entry["original_section_freeze"]))
    for pid in expected:
        proof = catalog[pid]["technical"]
        assert proof["mode"] == "review"
        pairs.append((pid, pid, proof, canonical_catalog[pid], proof["original_source_freeze"]))
    rows = []
    for sid, parent, proof, page, old_freeze in pairs:
        original = helper.read_ref(proof["original_report"])
        current = helper.read_ref(proof["report"])
        task = current["reviewer_task"]
        assigned = assignments[task]
        assert original["reviewer_task"] == task and task not in manifest["author_tasks"] and task not in readers
        assert assigned["role"] == "technical" and assigned["context"] == "fresh" and assigned["lesson_id"] == sid
        assert current["figure_sha256"] == page["figures_sha256"]
        body = helper.frozen_body(page)
        if sid != parent:
            assert expected[parent][sid] == body
        errors = validator._validate(ROOT, ROOT / page["source"], sid, body, current, owners, readers)
        assert not errors, (sid, errors)
        helper.check_original(original, old_freeze, sid, parent, task, validator, api,
                              original_source_snapshots=proof.get("original_source_snapshots", []))
        helper.check_report_pair(api, sealed, proof, original, current, assigned)
        if sid == parent:
            assert any(item.get("path") == manifest["numbered_section_reviews"]["path"] and item.get("sha256") == manifest["numbered_section_reviews"]["sha256"] for item in current["artifacts"])
        else:
            original_issues = set(api.reported_issues(original))
            closure = current.get("original_issue_resolutions", [])
            assert len(closure) == len(original_issues) and {item["json_pointer"] for item in closure} == original_issues
        rows.append({"lesson_id": sid, "canonical_parent": parent, "reviewer_task": task,
                     "original_report": proof["original_report"], "current_report": proof["report"],
                     "original_freeze": old_freeze, "original_source_snapshots": proof.get("original_source_snapshots", []),
                     "source_sha256": page["source_sha256"], "figure_sha256": page["figures_sha256"],
                     "original_collection": proof["original_collection"], "current_collection": proof["current_collection"],
                     "native_original": "accepted", "native_current": "accepted", "native_report_pair": "accepted"})
    # Exact issue index binding is checked without claiming final disposition approval.
    helper.check_section_issues(manifest, bundle, {entry["lesson_id"]: entry for entry in bundle["sections"]}, section_reports, catalog, api, intake=True)
    # Necessary negatives against the REAL completed stage and actual original/current data.
    w1 = next(entry for entry in bundle["sections"] if entry["lesson_id"] == "W.1")
    w1_original, w1_current = helper.read_ref(w1["original_report"]), helper.read_ref(w1["report"])
    w1_assignment = assignments[w1_current["reviewer_task"]]
    negatives = []
    for case in ("missing_current_row", "duplicate_current_row", "receipt_time_mismatch", "report_not_sealed", "creation_after_stage", "callback_window_after_stage", "original_current_receipt_reversed"):
        copied = copy.deepcopy(sealed)
        proof = copy.deepcopy(w1)
        current = copy.deepcopy(w1_current)
        phase = copied["omni-mainline-01"]["technical"]
        match = next(i for i, row in enumerate(phase["collection_index"]) if row["path"] == proof["report"]["path"] and row["sha256"] == proof["report"]["sha256"])
        if case == "missing_current_row":
            del phase["collection_index"][match]
        elif case == "duplicate_current_row":
            phase["collection_index"].append(copy.deepcopy(phase["collection_index"][match]))
        elif case == "receipt_time_mismatch":
            phase["collection_index"][match]["received_at"] = "2026-10-29T18:00:00+00:00"
        elif case == "report_not_sealed":
            phase["report_sha256s"].remove(proof["report"]["sha256"])
        elif case == "creation_after_stage":
            current["created_at"] = "2026-10-29T20:00:00+00:00"
        elif case == "callback_window_after_stage":
            proof["current_dispatch_window"] = {"not_before": "2026-10-29T20:00:00+00:00", "not_after": "2026-10-29T20:01:00+00:00"}
        elif case == "original_current_receipt_reversed":
            proof["current_collection"]["received_at"] = "2026-10-29T10:27:18+00:00"
            phase["collection_index"][match]["received_at"] = proof["current_collection"]["received_at"]
        negatives.append(attempt(case, lambda: helper.check_report_pair(api, copied, proof, w1_original, current, w1_assignment), False))
    t2 = next(entry for entry in bundle["sections"] if entry["lesson_id"] == "T.2")
    t2_original = helper.read_ref(t2["original_report"])
    for case in ("T2_missing_historical_snapshots", "T2_wrong_historical_snapshot_hash"):
        snapshots = [] if case == "T2_missing_historical_snapshots" else copy.deepcopy(t2["original_source_snapshots"])
        if snapshots:
            snapshots[0]["snapshot"]["sha256"] = "0" * 64
        negatives.append(attempt(case, lambda: helper.check_original(t2_original, t2["original_section_freeze"], "T.2", "training", t2_original["reviewer_task"], validator, api, original_source_snapshots=snapshots), False))
    for case in ("W1_current_stale_source", "W1_current_bad_artifact"):
        changed = copy.deepcopy(w1_current)
        if case == "W1_current_stale_source":
            changed["source_sha256"] = "0" * 64
        else:
            changed["artifacts"][0]["sha256"] = "0" * 64
        page = section_catalog["W.1"]
        errors = validator._validate(ROOT, ROOT / page["source"], "W.1", helper.frozen_body(page), changed, owners, readers)
        negatives.append({"case": case, "accepted": not errors, "expected_accept": False, "matches_expectation": bool(errors), "errors": errors})
    output = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": TASK,
              "kind": "bounded_two_original_issue_real_sealed_receipt_recheck_not_expanded_audit",
              "command": "uv run --extra cpu python " + Path(__file__).relative_to(ROOT).as_posix(),
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; read-only actual reports and deep-copy negatives"},
              "helper": ref(helper_path), "native_api": ref(ROOT / "docs/review-tools/incremental_revision.py"),
              "native_validator": ref(ROOT / "scripts/check_technical_reviews.py"),
              "packet": ref(packet_path), "manifest": ref(manifest_path), "actual_seal": ref(seal_path),
              "actual_seal_started_at": seal["started_at"], "actual_seal_completed_at": seal["completed_at"],
              "seal_formal_row_count": len(row_keys), "source_freeze": manifest["source_freeze"],
              "section_freeze": bundle["section_freeze"], "bundle": manifest["numbered_section_reviews"],
              "issue_index": bundle["issue_inventory"], "assignments": manifest["assignments"],
              "fresh_rerun_retained_61_code_probes": rerun_code, "actual_26_plus_4_native_results": rows,
              "necessary_negative_cases": negatives,
              "summary": {"native_original_passes": len(rows), "native_current_passes": len(rows), "native_sealed_report_pair_passes": len(rows),
                          "section_count": 26, "whole_count": 4, "code_probe_count": 61, "negative_case_count": len(negatives),
                          "unexpected": [item for item in negatives if not item["matches_expectation"]],
                          "exact_issue_index_binding": "passed intake schema binding; final root dispositions not certified"},
              "scope": "Only original TOOLS-ORIGINAL-SCHEMA-01 and TOOLS-W1-INTEGRATION-PENDING-02. Genuine completed technical seal used without edits. This output and resulting callback occur after that seal, for later continuity intake; never backfill these outputs into technical01. No full210+ scope, global gate, all100canonical semantics, historical tool-group fallback, reader/coherence/DP or publication approval."}
    target = OUT / "sealed-local-fix-results-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": ref(target), "summary": output["summary"], "actual_seal": output["actual_seal"], "helper": output["helper"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
