"""限單次失敗且不退費；真原receipt固定，官方terminal/history必須同時成立。"""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from scripts.omni_revision import preflight
from scripts.omni_revision import reconcile_failed_probe as reconciliation
from scripts.omni_revision.finance import digest, reservation_identity

NOW = datetime(2026, 10, 29, 21, 33, tzinfo=UTC)


def original_state():
    path = reconciliation.ROOT / "docs/course-revision-20261029-omni/infrastructure/failed-probe-attempt-v1.json"
    return {
        "attempts": [json.loads(path.read_bytes())],
        "reserved_total_usd": "0.40",
        "origin_unchanged_fixture": "retained",
    }


def inspection():
    original = original_state()["attempts"][0]
    return {
        "observed_at": NOW.isoformat(),
        "workspace": "birdhackor",
        "environment": "main",
        "attempt_id": reconciliation.ATTEMPT,
        "reservation_sha256": reconciliation.RESERVATION_SHA256,
        "ownership": json.loads(original["purpose"])["ownership"],
        "remote_mutation": False,
        "rpc_allowlist": ["WorkspaceNameLookup", "AppGetInfo", "SandboxList"],
        "all_sandboxes_include_finished": [],
        "app_info": {
            "info": {
                "app_id": reconciliation.APP_ID,
                "description": "omni-probe-40957487215-1",
                "environment_name": "main",
                "lifecycle": {"app_state": "APP_STATE_STOPPED", "stopped_at": 1793309083.176543},
            }
        },
    }


def test_exact_evidence_closes_only_status_and_preserves_original_bound_receipt_identity():
    state = original_state()
    before = deepcopy(state)
    following = reconciliation.resolve(state, inspection(), NOW)
    old, new = state["attempts"][0], following["attempts"][0]
    assert state == before
    assert new["status"] == "failed-stopped" and new["receipt"]["success"] is False
    assert new["receipt"]["original_receipt"] == old["receipt"]
    assert digest(new["receipt"]["original_receipt"]) == reconciliation.ORIGINAL_RECEIPT_SHA256
    assert reservation_identity(new) == reservation_identity(old)
    assert following["reserved_total_usd"] == "0.40"
    assert following["origin_unchanged_fixture"] == "retained"


@pytest.mark.parametrize(
    "change",
    [
        "running",
        "wrong_app",
        "wrong_environment",
        "wrong_description",
        "has_finished_instance",
        "stale",
        "future",
        "stopped_before_attempt",
        "wrong_reservation",
        "mutation",
    ],
)
def test_missing_or_wrong_official_evidence_never_closes_unknown_resources(change):
    value = inspection()
    info = value["app_info"]["info"]
    if change == "running":
        info["lifecycle"]["app_state"] = "APP_STATE_EPHEMERAL"
    elif change == "wrong_app":
        info["app_id"] = "ap-other"
    elif change == "wrong_environment":
        info["environment_name"] = "other"
    elif change == "wrong_description":
        info["description"] = "someone-elses-app"
    elif change == "has_finished_instance":
        value["all_sandboxes_include_finished"] = [{"id": "sb-existing-even-if-finished"}]
    elif change == "stale":
        value["observed_at"] = (NOW - timedelta(seconds=121)).isoformat()
    elif change == "future":
        value["observed_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif change == "stopped_before_attempt":
        info["lifecycle"]["stopped_at"] = 1
    elif change == "wrong_reservation":
        value["reservation_sha256"] = "0" * 64
    else:
        value["remote_mutation"] = True
    with pytest.raises(ValueError):
        reconciliation.resolve(original_state(), value, NOW)


@pytest.mark.parametrize(
    "change", ["different_attempt", "different_receipt", "changed_bound", "other_attempts", "resolved_twice"]
)
def test_one_off_rule_cannot_be_reused_or_rewrite_history(change):
    state = original_state()
    original = state["attempts"][0]
    if change == "different_attempt":
        original["attempt_id"] = "probe-other"
    elif change == "different_receipt":
        original["receipt"]["error_type"] = "TimeoutError"
    elif change == "changed_bound":
        original["components_usd"]["compute"] = "0"
    elif change == "other_attempts":
        state["attempts"].append(deepcopy(original))
    else:
        original["status"] = "failed-stopped"
    with pytest.raises(ValueError):
        reconciliation.resolve(state, inspection(), NOW)


def test_actual_modal_json_keys_are_snake_case_and_display_headers_are_not_json():
    actual = {
        "app_id": reconciliation.APP_ID,
        "description": "omni-probe-40957487215-1",
        "state": "stopped",
        "tasks": "0",
    }
    assert preflight.stopped_app(actual)
    assert not preflight.stopped_app({**actual, "tasks": "1"})
    assert not preflight.stopped_app({"App ID": reconciliation.APP_ID, "State": "stopped", "Tasks": "0"})
