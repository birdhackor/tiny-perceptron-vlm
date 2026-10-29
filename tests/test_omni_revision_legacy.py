"""Old quota identity/history and read-only P1 failure boundaries."""

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from scripts.omni_revision import legacy, preflight
from scripts.selftrained import finance as old

NOW = datetime(2026, 10, 29, 12, tzinfo=UTC)


def legacy_fixture(monkeypatch):
    # Test the new relational verifier with small synthetic old-epoch evidence.
    # Its unchanged old-policy pin validator is checked separately below.
    monkeypatch.setattr(old, "validate_policy", lambda policy, policy_sha: policy)
    sha = "a" * 64
    snapshot = {
        "source": old.SOURCE,
        "workspace_name": "birdhackor",
        "observed_at": "2026-10-06T10:00:00+00:00",
        "summaries": [
            {
                "cycle": "2026-09",
                "start": "2026-09-01T00:00:00+00:00",
                "end": "2026-10-01T00:00:00+00:00",
                "metered_cost_usd": "0",
            },
            {
                "cycle": "2026-10",
                "start": "2026-10-01T00:00:00+00:00",
                "end": "2026-11-01T00:00:00+00:00",
                "metered_cost_usd": "7.25",
            },
        ],
    }
    carry_record = {
        "run_id": "before",
        "reserved_usd": "0.54",
        "status": "completed",
        "finished_at": "2026-10-06T11:00:00+00:00",
    }
    immutable = {k: v for k, v in carry_record.items() if k not in {"status", "finished_at", "receipt_sha256"}}
    policy = {
        "round_id": "old-round",
        "workspace_name": "birdhackor",
        "baseline": {"observed_at": "2026-10-06T09:00:00+00:00", "cycle_gross_usd": {"2026-09": "0", "2026-10": "6"}},
        "carry_in_bounds": [
            {"run_id": "before", "reserved_usd": "0.54", "reservation_immutable_sha256": old.canonical_sha(immutable)}
        ],
    }
    current = {
        "run_id": "old-current",
        "reserved_usd": "0.70",
        "status": "completed",
        "gross_quota_policy_sha256": sha,
        "billing_before": {"gross_quota_snapshot": snapshot},
    }
    ledger = {
        "gross_quota_policy_sha256": sha,
        "reserved_total_usd": "49.64",
        "reservations": [
            {"run_id": "history", "reserved_usd": "48.40", "status": "completed"},
            carry_record,
            current,
        ],
    }
    quota = {
        "schema": old.SCHEMA,
        "round_id": policy["round_id"],
        "policy_sha256": sha,
        "policy_canonical_sha256": old.canonical_sha(policy),
        "last_observed_at": snapshot["observed_at"],
        "month_gross_highwater_usd": {"2026-09": "0", "2026-10": "7.25"},
        "attempts": [
            {
                "run_id": "old-current",
                "reserved_usd": "0.70",
                "reserved_at": "2026-10-06T10:00:01+00:00",
                "billing_snapshot_sha256": old.canonical_sha(snapshot),
            }
        ],
    }
    return ledger, quota, policy, sha


def test_actual_committed_old_policy_keeps_its_unchanged_identity():
    raw = (preflight.ROOT / legacy.POLICY_PATH).read_bytes()
    policy = json.loads(raw)
    assert old.validate_policy(policy, hashlib.sha256(raw).hexdigest()) is policy


def test_only_quota_attempts_and_pinned_carry_are_retained_not_historical_audit(monkeypatch):
    report = legacy.verify(*legacy_fixture(monkeypatch), NOW)
    assert report["carry_usd"] == "1.24"
    assert report["historical_audit_total_usd"] == "49.64"
    assert {item["run_id"] for item in report["retained_bounds"]} == {"before", "old-current"}


@pytest.mark.parametrize(
    "change",
    [
        "empty-quota",
        "marker",
        "missing-id",
        "extra-id",
        "bound",
        "immutable",
        "snapshot",
        "highwater",
        "missing-month",
        "active",
    ],
)
def test_corrupt_old_relationships_or_missing_history_are_not_reconciled_implicitly(monkeypatch, change):
    ledger, quota, policy, sha = legacy_fixture(monkeypatch)
    if change == "empty-quota":
        quota = {}
    elif change == "marker":
        ledger["gross_quota_policy_sha256"] = "b" * 64
    elif change == "missing-id":
        quota["attempts"] = []
    elif change == "extra-id":
        ledger["reservations"].append(
            {"run_id": "extra", "reserved_usd": "0.10", "status": "completed", "gross_quota_policy_sha256": sha}
        )
    elif change == "bound":
        quota["attempts"][0]["reserved_usd"] = "0.01"
    elif change == "immutable":
        ledger["reservations"][1]["batch_id"] = "tampered"
    elif change == "snapshot":
        ledger["reservations"][2]["billing_before"]["gross_quota_snapshot"]["summaries"][1]["metered_cost_usd"] = "1"
    elif change == "highwater":
        quota["month_gross_highwater_usd"] = {}
    elif change == "missing-month":
        quota["month_gross_highwater_usd"].pop("2026-09")
    else:
        ledger["reservations"][2]["status"] = "running"
    with pytest.raises((ValueError, KeyError)):
        legacy.verify(ledger, quota, policy, sha, NOW)


def test_legacy_stored_highwater_is_reconstructed_from_actual_snapshots(monkeypatch):
    ledger, quota, policy, sha = legacy_fixture(monkeypatch)
    second_snapshot = deepcopy(ledger["reservations"][2]["billing_before"]["gross_quota_snapshot"])
    second_snapshot["observed_at"] = "2026-10-06T10:01:00+00:00"
    second_snapshot["summaries"][1]["metered_cost_usd"] = "7.10"
    ledger["reservations"].append(
        {
            "run_id": "after",
            "reserved_usd": "0.10",
            "status": "completed",
            "gross_quota_policy_sha256": sha,
            "billing_before": {"gross_quota_snapshot": second_snapshot},
        }
    )
    ledger["reserved_total_usd"] = "49.74"
    quota["attempts"].append(
        {
            "run_id": "after",
            "reserved_usd": "0.10",
            "reserved_at": "2026-10-06T10:01:01+00:00",
            "billing_snapshot_sha256": old.canonical_sha(second_snapshot),
        }
    )
    quota["last_observed_at"] = second_snapshot["observed_at"]
    assert legacy.verify(ledger, quota, policy, sha, NOW)["month_gross_highwater_usd"]["2026-10"] == "7.25"
    quota["month_gross_highwater_usd"]["2026-10"] = "7.10"
    with pytest.raises(ValueError, match="high-water"):
        legacy.verify(ledger, quota, policy, sha, NOW)


def test_readonly_inventory_preserves_invalid_raw_quota_and_cannot_initialize_from_a_terminal_word(
    tmp_path, monkeypatch
):
    ledger_raw = json.dumps(
        {"reservations": [{"run_id": "history", "status": "completed", "reserved_usd": "48.40"}]}
    ).encode()
    quota_raw = b"{}"

    class Volume:
        objects = SimpleNamespace(list=lambda **_: [])

        @staticmethod
        def from_name(name, create_if_missing):
            assert name == "tiny-perceptron-course" and create_if_missing is False
            return SimpleNamespace(read_file=lambda path: iter([ledger_raw if path == "budget.json" else quota_raw]))

    monkeypatch.setattr(preflight, "app_rows", lambda: [])
    result = preflight.inventory(SimpleNamespace(Volume=Volume), tmp_path)
    assert result["errors"] and "legacy_verification" not in result
    assert result["remote_compute_started"] is False and result["new_volume_created"] is False
    assert (tmp_path / "legacy-gross-quota.json").read_bytes() == quota_raw
