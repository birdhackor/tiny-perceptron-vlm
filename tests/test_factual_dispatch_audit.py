"""Dispatch audit regression fixtures are synthetic metadata, never course review reports."""

import importlib.util
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "docs/review-tools/audit_factual_round.py"
SPEC = importlib.util.spec_from_file_location("factual_dispatch_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture_record():
    task = "/root/test_coordinator/test_section"
    raw = json.dumps({"reviewer_task": task, "source_sha256": "fixture-body"}).encode()
    record = {
        "status": "pass",
        "reviewer_task": task,
        "current_source_sha256": "fixture-body",
        "report_sha256": audit.sha(raw),
        "dispatches": [{"reviewer_task": task, "fork_turns": "none"}],
    }
    return record, raw


def test_declared_unique_dispatch_matches_collected_version():
    record, raw = fixture_record()
    assert audit.dispatch_errors(record, raw, "/root/test_coordinator") == []


def test_changed_report_with_same_body_requires_real_collection():
    record, raw = fixture_record()
    changed = json.loads(raw)
    changed["additional_note"] = "a different report version"
    errors = audit.dispatch_errors(record, json.dumps(changed).encode(), "/root/test_coordinator")
    assert "collected identity or report digest is stale" in errors


def test_full_history_fork_is_not_a_fresh_dispatch():
    record, raw = fixture_record()
    record["dispatches"][0]["fork_turns"] = "all"
    errors = audit.dispatch_errors(record, raw, "/root/test_coordinator")
    assert "missing current round fork-none dispatch declaration" in errors


def test_void_dispatch_is_preserved_without_invalidating_a_clean_replacement():
    record, raw = fixture_record()
    record["dispatches"].insert(
        0,
        {
            "reviewer_task": "/root/test_coordinator/contaminated_section",
            "fork_turns": "none",
            "status": "void_contamination_before_verdict",
            "dispatched_at": None,
        },
    )
    assert audit.dispatch_errors(record, raw, "/root/test_coordinator") == []


def test_void_dispatch_cannot_be_the_current_passed_report_owner():
    record, raw = fixture_record()
    record["dispatches"][0]["status"] = "void_contamination_before_verdict"
    errors = audit.dispatch_errors(record, raw, "/root/test_coordinator")
    assert "current report is attributed to a void factual dispatch" in errors
