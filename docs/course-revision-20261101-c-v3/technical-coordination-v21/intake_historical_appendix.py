"""Receive a real same-owner bounded appendix while T remains OPEN.

Administrative parse/hash/native structure and identity checks only. This is not
an independent semantic judgment, completed historical gate, or T seal.
"""
import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/course-revision-20261101-c-v3/technical-coordination-v21"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("report", "packet", "assignment", "dispatch", "original-receipt",
                 "current-receipt", "receipt", "group-output"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    intake = module("history_appendix_intake_base", BASE / "intake_report.py")
    assert intake.ROOT == ROOT
    ref, doc, local = intake.reference, intake.document, intake.local
    packet_ref = ref(args.packet)
    packet = doc(packet_ref)
    assignment_ref = ref(args.assignment)
    assignment = doc(assignment_ref)
    original_receipt_ref, current_receipt_ref = ref(args.original_receipt), ref(args.current_receipt)
    original_receipt, current_receipt = doc(original_receipt_ref), doc(current_receipt_ref)
    task, pid, wave = assignment["task"], assignment["lesson_id"], assignment["wave"]
    assert packet["kind"] == "page" and packet["page_id"] == pid
    assert assignment["role"] == "technical" and assignment["context"] == "fresh"
    assert original_receipt["intake_kind"] == "original" and current_receipt["intake_kind"] == "current"
    for receipt in (original_receipt, current_receipt):
        assert receipt["reviewer_task"] == task and receipt["lesson_id"] == pid and receipt["wave"] == wave
    original_ref, full_ref = original_receipt["report"], current_receipt["report"]
    original, full = doc(original_ref), doc(full_ref)
    assert original["reviewer_task"] == full["reviewer_task"] == task
    assert not original.get("historical_issue_rechecks") and not full.get("historical_issue_rechecks")
    assert full["verdict"] == "pass"
    assert intake.instant(original_receipt["received_at"]) <= intake.instant(current_receipt["received_at"])
    dispatch_ref = ref(args.dispatch)
    dispatch = doc(dispatch_ref)
    assert dispatch["task"] == task and dispatch["lesson_id"] == pid and dispatch["wave"] == wave
    window = dispatch["appendix_dispatch_window"]
    assert intake.instant(window["not_before"]) <= intake.instant(window["not_after"]) <= datetime.now(UTC)
    report_ref = ref(args.report)
    assert report_ref not in (original_ref, full_ref)
    report = doc(report_ref)
    assert report["reviewer_task"] == task and report["lesson_id"] == pid
    assert intake.instant(current_receipt["received_at"]) <= intake.instant(report["created_at"]) <= datetime.now(UTC)
    assert intake.instant(window["not_before"]) <= intake.instant(report["created_at"])
    freeze = doc(intake.CANONICAL)
    page = next(p for p in freeze["pages"] if p["page_id"] == pid)
    assert report["source_sha256"] == page["source_sha256"] and report["figure_sha256"] == page["figures_sha256"]
    api = intake.module("history_appendix_api", "docs/review-tools/incremental_revision.py")
    setup = intake.module("history_appendix_setup", "scripts/omni_revision/review_setup.py")
    validator = intake.module("history_appendix_validator", "scripts/check_technical_reviews.py")
    setup.check_original(original, intake.CANONICAL, pid, pid, task, validator, api)
    for current in (full, report):
        errors = validator._validate(ROOT, local(page["source"]), pid, setup.frozen_body(page), current, {task: [pid]}, set())
        if current["verdict"] == "pass":
            assert not errors, errors
        else:
            assert current is report and current["verdict"] == "revise"
            setup.check_original(current, intake.CANONICAL, pid, pid, task, validator, api)
    expected = packet["historical_identities_only"]
    rows = report["historical_issue_rechecks"]
    assert len(rows) == len(expected) and len({r["issue_id"] for r in rows}) == len(rows)
    by_id = {r["issue_id"]: r for r in rows}
    assert set(by_id) == {r["issue_id"] for r in expected}
    source_ids = {r["id"] for r in report["sources"]}
    artifact_ids = {r["id"] for r in report["artifacts"]}
    for identity in expected:
        row = by_id[identity["issue_id"]]
        for field in ("disposition_ref", "original_report", "json_pointer", "historical_fix_verification"):
            assert row[field] == identity[field], (identity["issue_id"], field)
        disposition_ref = {k: row["disposition_ref"][k] for k in ("path", "sha256")}
        old = api.node(doc(disposition_ref), row["disposition_ref"]["json_pointer"])
        assert old["action"] == "fixed" and old["issue_id"] == row["issue_id"]
        assert row["original_report"] == old["report"] and row["json_pointer"] == old["json_pointer"]
        assert row["historical_fix_verification"] in old["verification"]
        api.node(doc(row["original_report"]), row["json_pointer"])
        api.node(doc(row["historical_fix_verification"]["report"]), row["historical_fix_verification"].get("json_pointer", ""))
        assert row["necessity"] == old["final_necessity"] == identity["original_necessity"]
        for field in ("owner", "reviewer_task"):
            if field in row:
                assert row[field] == task
        assert isinstance(row.get("reason"), str) and row["reason"].strip()
        assert row["current_source_sha256"] == page["source_sha256"] and row["current_figure_sha256"] == page["figures_sha256"]
        assert row["current_evidence"] and row["artifact_ids"] and set(row["artifact_ids"]) <= artifact_ids
        for evidence in row["current_evidence"]:
            assert evidence["source_id"] in source_ids and evidence["locator"].strip() and evidence["supports"].strip()
        if report["verdict"] == "pass":
            assert row["verdict"] == "pass" and row["status"] == "still_fixed"
    # Validate, but do not recreate or reinterpret, authenticated old refs.
    doc(packet["prior_active_pointer"])
    doc(packet["source_chain"])
    api.node(doc(packet["unavailability"]["report"]), packet["unavailability"]["json_pointer"])
    received = datetime.now(UTC).isoformat()
    receipt = {
        "schema_version": 1, "kind": "actual_technical_historical_appendix_intake",
        "collector_task": "/root/c_v3_technical_coord", "intake_kind": "historical_appendix",
        "report": report_ref, "reviewer_task": task, "lesson_id": pid,
        "created_at": report["created_at"], "received_at": received,
        "collection_role": "technical", "wave": wave, "assignment": assignment_ref,
        "appendix_dispatch": dispatch_ref, "dispatch_window": window,
        "original_report": original_ref, "full_review": full_ref,
        "original_receipt": original_receipt_ref, "current_receipt": current_receipt_ref,
        "packet": packet_ref, "source_freeze": intake.CANONICAL,
        "report_verdict": report["verdict"], "historical_issue_count": len(rows),
        "physical_root": str(ROOT),
        "validation_scope": "Actual parse/hash/native structure and exact issue/source identity; not independent semantic audit, completed historical gate or T seal."
    }
    output = local(args.receipt)
    assert output.is_relative_to(BASE)
    with output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    group = {
        "schema_version": 1, "kind": "actual_open_historical_group_merge_input",
        "recorded_at": datetime.now(UTC).isoformat(), "collector_task": receipt["collector_task"],
        "source_chain": packet["source_chain"], "source_group_index": packet["source_group_index"],
        "prior_active_pointer": packet["prior_active_pointer"], "receipt": ref(args.receipt),
        "group": {"kind": "page", "page_id": pid, "unavailability": packet["unavailability"],
                  "full_review": full_ref, "original_review": original_ref, "appendix": report_ref,
                  "collection": {"wave": wave, "collection_role": "technical", "received_at": received}},
        "note": "Preserves authenticated prior identity/unavailability. Parent owns final sidecar/wrapper and completed-stage validation."
    }
    output = local(args.group_output)
    assert output.is_relative_to(BASE)
    with output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(group, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"receipt": ref(args.receipt), "group": ref(args.group_output), "report": report_ref,
                      "received_at": received, "verdict": report["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
