"""Verify actual selected open receipts, never invent completed_at or real seal."""

import copy
import hashlib
import importlib.util
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
INDEX = "docs/course-revision-20261029-omni/reviews/technical-stage/whole-collectors-source14-index-v1.json"
TASK = "/root/omni_technical_coordinator/technical_review_tools"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    helper = load("bounded_receipt_helper", ROOT / "scripts/omni_revision/review_setup.py")
    api = load("bounded_receipt_api", ROOT / "docs/review-tools/incremental_revision.py")
    validator = load("bounded_receipt_validator", ROOT / "scripts/check_technical_reviews.py")
    index_path = ROOT / INDEX
    index = json.loads(index_path.read_bytes())
    assert sha(index_path.read_bytes()) == "ea7aa5c1ef580e7b48574952508d36760ec528a6dacb6bd6d55bfb23ef31a233"
    for key in ("source_freeze", "numbered_freeze", "bundle", "issues", "assignments", "actual_collection_snapshot", "raw_collection_snapshot"):
        api.artifact(ROOT, index[key])
    bundle = helper.read_ref(index["bundle"])
    phase = helper.read_ref(index["actual_collection_snapshot"])
    raw_phase = helper.read_ref(index["raw_collection_snapshot"])
    assignments = helper.read_ref(index["assignments"])["tasks"]
    task_map = {row["task"]: row for row in assignments}
    selected_rows = phase["collection_index"]
    raw_rows = raw_phase["collection_index"]
    assert phase["status"] == "in_progress" and "completed_at" not in phase
    assert phase["role"] == "technical" and phase["raw_collection_snapshot"] == index["raw_collection_snapshot"]
    originals = []
    rows = []
    pairs = []
    for entry in bundle["sections"]:
        pairs.append((entry["lesson_id"], {**entry}, entry["original_section_freeze"], None))
    for entry in index["whole_pages"]:
        proof = {**entry, "report": entry["current_report"]}
        pairs.append((entry["page_id"], proof, entry["original_source_freeze"], entry))
    assert len(pairs) == 30
    for pid, proof, original_freeze, whole_entry in pairs:
        original = helper.read_ref(proof["original_report"])
        current = helper.read_ref(proof["report"])
        task = current["reviewer_task"]
        assignment = task_map[task]
        assert original["reviewer_task"] == task
        assert assignment["role"] == "technical" and assignment["context"] == "fresh" and assignment["lesson_id"] == pid
        assert api.at(original["created_at"]) <= api.at(current["created_at"])
        assert api.at(proof["original_collection"]["received_at"]) <= api.at(proof["current_collection"]["received_at"])
        if whole_entry is not None:
            helper.check_original(original, original_freeze, pid, pid, task, validator, api,
                                  original_source_snapshots=proof.get("original_source_snapshots", []))
            originals.append({"page_id": pid, "original_report": proof["original_report"], "original_source_freeze": original_freeze,
                              "check_original": "accepted with actual original snapshots and original freeze", "current_self_check_ref_only": proof.get("current_self_check")})
        for role_key, report_key in (("original_collection", "original_report"), ("current_collection", "report")):
            receipt = proof[role_key]
            reference = proof[report_key]
            report = original if role_key == "original_collection" else current
            selected_matches = [row for row in selected_rows if row["path"] == reference["path"] and row["sha256"] == reference["sha256"]]
            raw_matches = [row for row in raw_rows if row["path"] == reference["path"] and row["sha256"] == reference["sha256"]]
            assert len(selected_matches) == len(raw_matches) == 1
            match = selected_matches[0]
            assert match == raw_matches[0] and match["received_at"] == receipt["received_at"]
            assert match["reviewer_task"] == task and match["lesson_id"] == pid
            assert match["wave"] == receipt["wave"] and match["collection_role"] == receipt["collection_role"] == "technical"
            assert match["created_at"] == report["created_at"]
            assert reference["sha256"] in phase["report_sha256s"]
            window = assignment["dispatch_window"] if role_key == "original_collection" else proof.get("current_dispatch_window", assignment["dispatch_window"])
            created, received = api.at(report["created_at"]), api.at(receipt["received_at"])
            assert api.at(phase["started_at"]) <= api.at(window["not_before"]) <= api.at(window["not_after"])
            assert api.at(window["not_before"]) <= created <= received <= api.at(index["created_at"]) <= datetime.now(UTC)
            rows.append({"page_id": pid, "role": role_key, "report": reference, "created_at": report["created_at"],
                         "received_at": receipt["received_at"], "unique_selected_and_raw": True, "owner_and_open_chronology_valid": True})
    # Native receipt checks are deliberately called with the TRUE OPEN phase.
    # Intact receipt cannot be sealed-accepted because completed_at does not exist.
    w1 = next(row for row in bundle["sections"] if row["lesson_id"] == "W.1")
    w1_report = helper.read_ref(w1["report"])
    w1_assignment = task_map[w1_report["reviewer_task"]]
    native_results = []
    cases = ("actual_open_pair_pending", "missing_current_row", "duplicate_current_row", "mismatched_current_receipt_time", "wrong_ref_hash", "reader_collection_for_technical")
    for case in cases:
        proof = copy.deepcopy(w1)
        copied_phase = copy.deepcopy(phase)
        if case in {"missing_current_row", "duplicate_current_row", "mismatched_current_receipt_time"}:
            matches = [i for i, row in enumerate(copied_phase["collection_index"]) if row["path"] == proof["report"]["path"] and row["sha256"] == proof["report"]["sha256"]]
            pos = matches[0]
            if case == "missing_current_row":
                del copied_phase["collection_index"][pos]
            elif case == "duplicate_current_row":
                copied_phase["collection_index"].append(copy.deepcopy(copied_phase["collection_index"][pos]))
            else:
                copied_phase["collection_index"][pos]["received_at"] = "2026-10-29T17:00:00+00:00"
        elif case == "wrong_ref_hash":
            proof["report"]["sha256"] = "0" * 64
        elif case == "reader_collection_for_technical":
            proof["current_collection"]["collection_role"] = "reader"
        try:
            if case == "actual_open_pair_pending":
                helper.check_report_pair(api, {phase["wave"]: {"technical": copied_phase}}, proof,
                                         helper.read_ref(proof["original_report"]), w1_report, w1_assignment)
            else:
                helper.check_receipt(api, {phase["wave"]: {"technical": copied_phase}}, proof["report"], w1_report,
                                     proof["current_collection"], proof.get("current_dispatch_window", w1_assignment["dispatch_window"]))
            native_results.append({"case": case, "accepted": True, "unexpected": True})
        except (ValueError, KeyError, TypeError, OSError) as error:
            native_results.append({"case": case, "accepted": False, "error": str(error),
                                   "expected_pending_not_failure": case == "actual_open_pair_pending" and isinstance(error, KeyError) and "completed_at" in str(error),
                                   "unexpected": False})
    output = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": TASK,
              "kind": "bounded_actual_open_collection_checks_not_native_sealed_approval",
              "command": "uv run --extra cpu python " + Path(__file__).relative_to(ROOT).as_posix(),
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; read-only actual selected collection"},
              "immutable_index": ref(index_path), "selected_collection": index["actual_collection_snapshot"], "raw_collection": index["raw_collection_snapshot"],
              "assignments": index["assignments"], "bundle": index["bundle"], "issue_index": index["issues"],
              "phase_status": phase["status"], "phase_has_completed_at": "completed_at" in phase, "selected_unique_rows": len(selected_rows),
              "pair_count": len(pairs), "original_current_references_checked": len(rows), "receipt_rows": rows,
              "four_whole_original_checks": originals, "native_open_phase_and_negative_checks": native_results,
              "summary": {"all_30_original_current_pairs_actually_received": True, "all_60_references_have_unique_selected_and_raw_receipt": True,
                          "native_report_pair_completion": "pending: actual phase lacks completed_at", "unexpected": [row for row in native_results if row.get("unexpected")]},
              "scope": "55 unique selected actual receipt rows cover 60 original/current references for 26 sections plus four collectors; raw snapshot is larger and not declared fully audited. No completed_at added, no real seal synthesized. Open-receipt facts pass independent checks; native sealed traversal remains pending."}
    target = OUT / "local-fix-receipts-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": ref(target), "summary": output["summary"], "native_receipt_cases": native_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
