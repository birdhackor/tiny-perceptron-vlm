"""Budget failure boundaries, not a replay of the implementation's happy path."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from scripts.omni_revision.finance import (
    POLICY,
    SOURCE,
    GitHubLedger,
    digest,
    finish,
    new_state,
    reserve,
)

NOW = datetime(2026, 10, 29, 12, tzinfo=UTC)
REVISION = "a" * 40


def snapshot(now=NOW, october="8.00", november=None):
    rows = [
        {
            "cycle": "2026-10",
            "start": "2026-10-01T00:00:00+00:00",
            "end": "2026-11-01T00:00:00+00:00",
            "metered_cost_usd": october,
        }
    ]
    if november is not None:
        rows.append(
            {
                "cycle": "2026-11",
                "start": "2026-11-01T00:00:00+00:00",
                "end": "2026-12-01T00:00:00+00:00",
                "metered_cost_usd": november,
            }
        )
    return {"source": SOURCE, "workspace": "birdhackor", "observed_at": now.isoformat(), "summaries": rows}


def inventory(now=NOW):
    report = {
        "retained_bounds": [{"run_id": "old-1", "status": "completed", "reserved_usd": "0.50"}],
        "carry_usd": "0.50",
        "historical_audit_total_usd": "48.40",
        "historical_audit_excluded_from_carry": True,
        "month_gross_highwater_usd": {"2026-10": "7.50"},
    }
    reconciliation = {
        "schema": "omni-legacy-reconciliation-v1",
        "reviewer_task": "/root/independent_legacy_reviewer",
        "ledger_sha256": "b" * 64,
        "quota_sha256": "c" * 64,
        "legacy_policy_sha256": "d" * 64,
        "legacy_verification_sha256": digest(report),
        "existing_volumes_sha256": digest([]),
        "retained_bounds": [
            {
                "run_id": "old-1",
                "reserved_usd": "0.50",
                "stop_evidence": [{"path": "docs/stopped.json", "sha256": "e" * 64}],
            }
        ],
    }
    return {
        "workspace": "birdhackor",
        "environment": "main",
        "observed_at": now.isoformat(),
        "ledger_sha256": "b" * 64,
        "quota_sha256": "c" * 64,
        "legacy_policy_sha256": "d" * 64,
        "legacy_verification": report,
        "existing_volumes": [],
        "reconciliation": reconciliation,
        "unknown_attempts": [],
        "active_apps": [],
        "errors": [],
    }


def components(total="1.00"):
    return {"compute": total, "startup_idle": "0", "build": "0", "egress": "0", "storage_cleanup": "0"}


def initialized():
    return new_state(snapshot(), inventory(), REVISION, now=NOW)


def reserved(state=None, amount="1.00", attempt="one", observed=None, now=NOW):
    return reserve(
        state or initialized(),
        observed or snapshot(),
        inventory(now),
        attempt,
        components(amount),
        REVISION,
        "bounded test",
        now=now,
    )


def terminal(state, attempt="one", success=True):
    return finish(
        state,
        attempt,
        {
            "success": success,
            "compute_stopped": True,
            "resources_verified": True,
            "cleanup_checks": {"own_app_stopped": True, "temporary_objects_cleared": True},
        },
    )


def test_completion_and_failure_keep_the_entire_bound_and_never_reuse_attempt():
    for success in (True, False):
        state = terminal(reserved(amount="59"), success=success)
        assert state["reserved_total_usd"] == "59"
        with pytest.raises(ValueError, match="reused"):
            reserved(state, attempt="one")
        assert reserved(state, amount="0.50", attempt="two")["reserved_total_usd"] == "59.50"
        with pytest.raises(ValueError, match="exceed USD60"):
            reserved(state, amount="1.01", attempt="two")


def test_unresolved_compute_or_storage_blocks_the_next_paid_attempt():
    running = reserved()
    with pytest.raises(ValueError, match="running or unresolved"):
        reserved(running, attempt="two")
    failed_cleanup = finish(running, "one", {"success": False, "compute_stopped": True, "volume_deleted": False})
    assert failed_cleanup["attempts"][0]["status"] == "unresolved"
    with pytest.raises(ValueError, match="running or unresolved"):
        reserved(failed_cleanup, attempt="two")


@pytest.mark.parametrize("age", [121, -1])
def test_official_snapshot_must_be_fresh_and_not_from_the_future(age):
    when = NOW - timedelta(seconds=age)
    with pytest.raises(ValueError, match="stale or future"):
        reserved(observed=snapshot(when))


def test_unknown_legacy_and_active_other_apps_are_not_terminated_implicitly():
    for field, value in (("unknown_attempts", [{"run_id": "old"}]), ("active_apps", [{"State": "running"}])):
        old = inventory()
        old[field] = value
        with pytest.raises(ValueError, match="reconciled"):
            new_state(snapshot(), old, REVISION, now=NOW)


def test_legacy_raw_files_cannot_change_during_this_round():
    old = inventory()
    old["quota_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="Legacy .*fingerprints"):
        reserve(initialized(), snapshot(), old, "new", components(), REVISION, "test", now=NOW)


def test_new_reservation_cannot_cover_an_existing_unexplained_workspace_increment():
    # Before the first reservation only the old pending USD0.50 is explained.
    with pytest.raises(ValueError, match="Unexplained"):
        reserved(amount="30", observed=snapshot(october="8.51"))
    assert reserved(observed=snapshot(october="8.50"))["reserved_total_usd"] == "1.00"


def test_calendar_rollover_preserves_october_and_rejects_missing_cycles_or_highwater_rollback():
    state = terminal(reserved())
    november = datetime(2026, 11, 1, tzinfo=UTC)
    observed = snapshot(november, october="8.1", november="0.1")
    updated = reserved(state, attempt="two", observed=observed, now=november)
    assert updated["month_gross_highwater_usd"] == {"2026-10": "8.1", "2026-11": "0.1"}
    with pytest.raises(ValueError, match="coverage"):
        reserved(state, attempt="two", observed=snapshot(november), now=november)
    with pytest.raises(ValueError, match="high-water"):
        reserved(state, attempt="two", observed=snapshot(october="7.99"))


def test_policy_tampering_inconsistent_totals_and_unpriced_boundaries_fail_closed():
    altered = initialized()
    altered["policy"] = deepcopy(POLICY)
    altered["policy"]["additional_gross_cap_usd"] = "600"
    with pytest.raises(ValueError, match="policy"):
        reserved(altered)
    altered = initialized()
    altered["reserved_total_usd"] = "1"
    with pytest.raises(ValueError, match="high-water/total"):
        reserved(altered)
    incomplete = components()
    del incomplete["storage_cleanup"]
    with pytest.raises(ValueError, match="boundary"):
        reserve(initialized(), snapshot(), inventory(), "one", incomplete, REVISION, "test", now=NOW)


@pytest.mark.parametrize("amount", [True, "NaN", "Infinity", "-1"])
def test_invalid_monetary_values_cannot_enter_the_ledger(amount):
    with pytest.raises(ValueError):
        reserved(amount=amount)


def test_official_increment_and_retained_bounds_are_both_charged_against_the_cap():
    state = terminal(reserved(amount="59"))
    with pytest.raises(ValueError, match="Retained official increment"):
        reserved(state, amount="1", attempt="two", observed=snapshot(october="67"))


def test_historical_audit_total_is_not_legacy_unreconciled_carry():
    state = initialized()
    assert state["origin"]["legacy_carry_usd"] == "0.50"
    assert state["origin"]["legacy_inventory"]["legacy_verification"]["historical_audit_total_usd"] == "48.40"
    old = inventory()
    old["legacy_verification"]["carry_usd"] = "48.40"
    old["reconciliation"]["legacy_verification_sha256"] = digest(old["legacy_verification"])
    with pytest.raises(ValueError, match="exact bound list"):
        new_state(snapshot(), old, REVISION, now=NOW)


def test_deleted_highwater_or_lowered_baseline_cannot_remove_official_history():
    state = terminal(reserved())
    state["month_gross_highwater_usd"] = {}
    with pytest.raises(ValueError, match="high-water/total"):
        reserved(state, attempt="two", observed=snapshot(october="7"))
    altered = initialized()
    altered["origin"]["baseline"]["summaries"][0]["metered_cost_usd"] = "0"
    with pytest.raises(ValueError, match="Initialization baseline"):
        reserved(altered)


def test_new_baseline_cannot_reset_retained_old_monthly_highwater():
    with pytest.raises(ValueError, match="legacy monthly high-water"):
        new_state(snapshot(october="7.49"), inventory(), REVISION, now=NOW)


def test_changed_persistent_volume_inventory_is_not_silently_authorized():
    old = inventory()
    old["existing_volumes"] = [{"name": "unexpected", "object_id": "vo-new"}]
    with pytest.raises(ValueError, match="persistent storage"):
        reserve(initialized(), snapshot(), old, "one", components(), REVISION, "test", now=NOW)


def test_terminal_requires_all_explicit_resource_checks_not_just_app_stopped():
    incomplete = finish(
        reserved(),
        "one",
        {
            "success": False,
            "compute_stopped": True,
            "resources_verified": True,
            "cleanup_checks": {"temporary_objects_cleared": False},
        },
    )
    assert incomplete["attempts"][0]["status"] == "unresolved"
    with pytest.raises(ValueError, match="running or unresolved"):
        reserved(incomplete, attempt="two")


def test_reconciliation_must_cover_every_old_bound_and_stop_evidence():
    for change in ("missing-bound", "missing-stop", "not-reviewed"):
        old = inventory()
        if change == "missing-bound":
            old["reconciliation"]["retained_bounds"] = []
        elif change == "missing-stop":
            old["reconciliation"]["retained_bounds"][0]["stop_evidence"] = []
        else:
            old["reconciliation"] = {}
        with pytest.raises(ValueError):
            new_state(snapshot(), old, REVISION, now=NOW)


def test_github_write_uses_cas_and_a_losing_controller_cannot_overwrite_a_winner(monkeypatch):
    store = GitHubLedger()
    current = {"blob": "before", "state": initialized()}

    def api(endpoint, payload=None, allow_missing=False):
        assert payload["sha"] == "before"
        if current["blob"] != payload["sha"]:
            raise RuntimeError("HTTP409 compare-and-swap conflict")
        current.update(blob="after", state="winner")
        return {"commit": {"sha": "commit"}, "content": {"sha": "after"}}

    monkeypatch.setattr(store, "api", api)
    assert store.write(reserved(attempt="a"), "before")["blob"] == "after"
    with pytest.raises(RuntimeError, match="compare-and-swap"):
        store.write(reserved(attempt="b"), "before")


def test_initialize_cannot_reset_an_existing_branch(monkeypatch):
    store = GitHubLedger()
    monkeypatch.setattr(store, "api", lambda *args, **kwargs: {"ref": "already exists"})
    with pytest.raises(ValueError, match="never reset"):
        store.initialize(initialized(), REVISION)
