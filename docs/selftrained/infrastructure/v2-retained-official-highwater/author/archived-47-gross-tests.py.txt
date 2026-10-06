"""Genuine narrow bookkeeping/runner checks; no SDK account or compute call."""

import ast
import importlib.util
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 6, 9, 30, tzinfo=UTC)
SHA = "a" * 64
RATES = {
    "cpu_hour_cost": "0.04730",
    "mem_gib_hour_cost": "0.00800",
    "gpu_hour_cost_l4": "0.80000",
    "egress_gib_cost": "0",
}


def module(name):
    spec = importlib.util.spec_from_file_location(
        "scripts.selftrained." + name, ROOT / "scripts/selftrained" / (name + ".py")
    )
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


finance = module("finance")
runner = module("modal_runner")


def fixture():
    carry = {"run_id": "gha-37441599016-1", "reserved_usd": "0.54", "status": "completed", "job_sha256": "e" * 64}
    policy = {
        "schema": finance.SCHEMA,
        "round_id": "20261006-additional30",
        "workspace_name": "birdhackor",
        "baseline": {
            "observed_at": "2026-10-06T09:15:32.796484+00:00",
            "gross_usd": "6.32341211",
            "cycle_gross_usd": {"2026-09": "0", "2026-10": "6.32341211"},
            "raw_sha256": finance.BASELINE_RAW_SHA256,
        },
        "additional_quota_usd": "30",
        "snapshot_max_age_seconds": 120,
        "carry_in_bounds": [
            {
                "run_id": "gha-37441599016-1",
                "reserved_usd": "0.54",
                "reservation_sha256": "c" * 64,
                "reservation_immutable_sha256": finance.canonical_sha(
                    {k: v for k, v in carry.items() if k != "status"}
                ),
            }
        ],
    }
    ledger = {
        "budget_usd": "55",
        "reserved_total_usd": "48.40",
        "reservations": [{"run_id": "old-history", "reserved_usd": "47.86", "status": "failed"}, carry],
    }
    snapshot = {
        "source": finance.SOURCE,
        "workspace_name": "birdhackor",
        "observed_at": NOW.isoformat(),
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
                "metered_cost_usd": "6.41341211",
                "billed_cost_usd": "0",
                "credits_usd": "-900",
            },
        ],
    }
    return policy, ledger, snapshot


def check(policy=None, ledger=None, snapshot=None, state=None, amount="0.54", run_id="gha-new", now=NOW):
    fixture_policy, fixture_ledger, fixture_snapshot = fixture()
    return finance.check_quota(
        policy or fixture_policy,
        SHA,
        state,
        snapshot or fixture_snapshot,
        ledger or fixture_ledger,
        amount,
        run_id,
        now,
    )


def record(ledger, state, status="completed"):
    item = state["attempts"][-1]
    ledger["reservations"].append(
        {
            "run_id": item["run_id"],
            "reserved_usd": item["reserved_usd"],
            "status": status,
            "gross_quota_policy_sha256": SHA,
            "billing_before": {"gross_quota_snapshot": fixture()[2]},
        }
    )


def test_exact_baseline_gross_credits_and_legacy_not_spent():
    state, report = check()
    assert report["authorized_gross_ceiling_usd"] == "36.32341211"
    assert Decimal(report["actual_gross_increment_usd"]) == Decimal(".09")
    assert Decimal(report["guarded_increment_usd"]) == Decimal("1.17")
    assert Decimal(state["attempts"][0]["reserved_usd"]) == Decimal(".54")
    assert "48.40" not in json.dumps(report) and "credits" not in json.dumps(state)


def test_additive_known_actual_other_project_plus_unmetered_attempt_rejects_unsafe_max():
    p, ledger, snapshot = fixture()
    state, _ = check(ledger=ledger, amount="1", run_id="gha-prior")
    record(ledger, state)
    snapshot["summaries"][-1]["metered_cost_usd"] = "35.82341211"  # delta29.5, possibly other project
    with pytest.raises(RuntimeError, match="exhausted"):
        check(ledger=ledger, snapshot=snapshot, state=state, amount=".1")
    assert max(Decimal("29.5"), Decimal("1.64")) < 30  # max formula would incorrectly accept


@pytest.mark.parametrize("status", ["completed", "failed", "failed-or-cancelled"])
def test_completed_and_failed_bounds_remain_and_exact_next_attempt(status):
    p, ledger, snapshot = fixture()
    state, _ = check(ledger=ledger, amount=".54", run_id="gha-first")
    record(ledger, state, status)
    later, report = check(ledger=ledger, state=state, run_id="gha-next")
    assert len(later["attempts"]) == 2
    assert Decimal(report["unreconciled_round_bounds_usd"]) == Decimal(".54")
    assert Decimal(report["guarded_increment_usd"]) == Decimal("1.71")


@pytest.mark.parametrize(
    "change",
    ["delete", "shrink", "unknown", "carry_overlap", "duplicate", "future_time", "wrong_round", "wrong_policy"],
)
def test_sidecar_counter_tamper_rejected(change):
    p, ledger, snapshot = fixture()
    state, _ = check(ledger=ledger, amount="1", run_id="gha-first")
    record(ledger, state)
    if change == "delete":
        state["attempts"].clear()
    elif change == "shrink":
        state["attempts"][0]["reserved_usd"] = ".01"
    elif change == "unknown":
        state["attempts"][0]["run_id"] = "gha-unknown"
    elif change == "carry_overlap":
        state["attempts"][0]["run_id"] = "gha-37441599016-1"
    elif change == "duplicate":
        state["attempts"].append(deepcopy(state["attempts"][0]))
    elif change == "future_time":
        state["attempts"][0]["reserved_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif change == "wrong_round":
        state["round_id"] = "reset-round"
    elif change == "wrong_policy":
        state["policy_sha256"] = "d" * 64
    with pytest.raises((RuntimeError, ValueError)):
        check(ledger=ledger, state=state)


@pytest.mark.parametrize(
    "change",
    [
        "workspace",
        "stale",
        "future",
        "naive_time",
        "before_baseline",
        "missing_month",
        "duplicate_month",
        "bad_month_end",
        "nan",
        "negative",
        "gross_regression",
    ],
)
def test_official_snapshot_invalid_fails_closed(change):
    p, ledger, snapshot = fixture()
    if change == "workspace":
        snapshot["workspace_name"] = "other-workspace"
    elif change == "stale":
        snapshot["observed_at"] = (NOW - timedelta(seconds=121)).isoformat()
    elif change == "future":
        snapshot["observed_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif change == "naive_time":
        snapshot["observed_at"] = "2026-10-06T09:30:00"
    elif change == "before_baseline":
        snapshot["observed_at"] = "2026-10-06T09:14:00+00:00"
    elif change == "missing_month":
        snapshot["summaries"].pop(0)
    elif change == "duplicate_month":
        snapshot["summaries"].append(deepcopy(snapshot["summaries"][0]))
    elif change == "bad_month_end":
        snapshot["summaries"][-1]["end"] = "2026-11-02T00:00:00+00:00"
    elif change in ("nan", "negative", "gross_regression"):
        snapshot["summaries"][-1]["metered_cost_usd"] = {"nan": "NaN", "negative": "-1", "gross_regression": "6.30"}[
            change
        ]
    with pytest.raises((RuntimeError, ValueError)):
        check(snapshot=snapshot)


def test_month_rollover_reads_carry_previous_month_and_retains_regression_highwater():
    p, ledger, snapshot = fixture()
    state, _ = check(ledger=ledger, run_id="gha-first")
    record(ledger, state)
    later = datetime(2026, 11, 1, 0, 2, tzinfo=UTC)
    snapshot["observed_at"] = later.isoformat()
    snapshot["summaries"].append(
        {
            "cycle": "2026-11",
            "start": "2026-11-01T00:00:00+00:00",
            "end": "2026-12-01T00:00:00+00:00",
            "metered_cost_usd": ".20",
        }
    )
    _, report = check(snapshot=snapshot, ledger=ledger, state=state, now=later)
    assert Decimal(report["actual_gross_increment_usd"]) == Decimal(".29")
    snapshot["summaries"][1]["metered_cost_usd"] = "6.40"
    next_state, retained = check(snapshot=snapshot, ledger=ledger, state=state, now=later)
    assert Decimal(retained["actual_gross_increment_usd"]) == Decimal(".27658789")
    assert Decimal(retained["retained_highwater_gross_increment_usd"]) == Decimal(".29")
    assert retained["guarded_increment_usd"] == report["guarded_increment_usd"]
    assert next_state["month_gross_highwater_usd"]["2026-10"] == "6.41341211"


def test_same_month_time_reverse_and_run_reuse_rejected():
    p, ledger, snapshot = fixture()
    state, _ = check(ledger=ledger, run_id="gha-first")
    record(ledger, state)
    with pytest.raises(ValueError, match="fresh unique"):
        check(ledger=ledger, state=state, run_id="gha-first")
    snapshot["observed_at"] = (NOW - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="backwards"):
        check(ledger=ledger, state=state, snapshot=snapshot)


def test_active_carry_or_attempt_blocks_new_paid_overlap():
    p, ledger, snapshot = fixture()
    ledger["reservations"][1]["status"] = "reserved"
    with pytest.raises(RuntimeError, match="no-overlap"):
        check(ledger=ledger)
    ledger["reservations"][1]["status"] = "completed"
    state, _ = check(ledger=ledger, run_id="gha-first")
    record(ledger, state, "reserved")
    with pytest.raises(RuntimeError, match="no-overlap"):
        check(ledger=ledger, state=state)


def test_carry_immutable_job_identity_cannot_restamp_amount_only():
    p, ledger, snapshot = fixture()
    ledger["reservations"][1]["job_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="immutable"):
        check(ledger=ledger)


def test_public_summary_api_only_missing_api_propagates_and_never_uses_credits():
    p, ledger, snapshot = fixture()
    called = []

    def summary(cycle):
        called.append(cycle)
        row = next(x for x in snapshot["summaries"] if x["cycle"] == cycle)
        return SimpleNamespace(
            start=finance.utc(row["start"]),
            end=finance.utc(row["end"]),
            metered_cost=Decimal(row["metered_cost_usd"]),
            billed_cost=Decimal(0),
            adjustments={"Credits": Decimal("-900")},
        )

    w = SimpleNamespace(name="birdhackor", billing=SimpleNamespace(summary=summary))
    output = finance.official_snapshot(w, p, SHA, NOW)
    assert called == ["2026-09", "2026-10"]
    assert "Credits" not in json.dumps(output) and "billed" not in json.dumps(output)

    def fail(cycle):
        raise PermissionError("synthetic API denied")

    w.billing.summary = fail
    with pytest.raises(PermissionError):
        finance.official_snapshot(w, p, SHA, NOW)


@pytest.fixture
def runner_time(monkeypatch):
    original = finance.check_quota
    monkeypatch.setattr(finance, "check_quota", lambda *args, **kwargs: original(*args, **kwargs, now=NOW))


def reserve(ledger, policy=None, state=None, snapshot=None, run_id="gha-new"):
    job = {"schema_version": 1, "stage": "pretrain"}
    if policy is not None:
        job["gross_quota_policy"] = {"path": "docs/selftrained/gross-quota.json", "sha256": SHA}
    return runner.reserve_entry(
        ledger,
        run_id,
        "selftrained-v2",
        "d" * 40,
        "e" * 64,
        job,
        "f" * 64,
        {"rates": RATES, "gross_quota_snapshot": snapshot or fixture()[2]},
        {},
        policy,
        SHA if policy else None,
        state,
    )


def test_runner_old_default_exact_and_new_history_survives_past55(runner_time):
    policy, ledger, snapshot = fixture()
    old = deepcopy(ledger["reservations"])
    entry, state = reserve(ledger, policy)
    assert ledger["reservations"][:2] == old
    assert Decimal(entry["gross_quota_guard"]["guarded_increment_usd"]) == Decimal(".92")
    runner.finish_entry(ledger, entry["run_id"], "failed")
    ledger["reserved_total_usd"] = "56.50"  # opaque old audit high-water, never spent or refunded
    entry2, state2 = reserve(ledger, policy, state, run_id="gha-next")
    assert Decimal(entry2["gross_quota_guard"]["unreconciled_round_bounds_usd"]) == Decimal(".29")
    assert Decimal(ledger["reserved_total_usd"]) == Decimal("56.79")
    with pytest.raises(ValueError, match="explicit"):
        reserve(ledger, run_id="gha-missing-policy")
    default = {
        "reserved_total_usd": "48.40",
        "reservations": [{"run_id": "history", "status": "failed", "reserved_usd": "48.40"}],
    }
    assert isinstance(reserve(default), dict)


def remote_fixture(tmp_path, monkeypatch):
    policy, ledger, snapshot = fixture()
    events = []
    monkeypatch.setattr(runner, "LEDGER", tmp_path / "budget.json")
    monkeypatch.setattr(runner, "GROSS_QUOTA", tmp_path / "gross-quota.json")
    runner.write_json(runner.LEDGER, ledger)
    tree = ast.parse((ROOT / "scripts/selftrained/modal_runner.py").read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "reserve_remote")
    function.decorator_list = []
    code = compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        "<actual candidate reserve_remote AST>",
        "exec",
    )
    scope = dict(vars(runner))
    scope["volume"] = SimpleNamespace(reload=lambda: events.append("reload"), commit=lambda: events.append("commit"))
    scope["prepared_records_gate"] = lambda *args: None
    workspace = SimpleNamespace(name="birdhackor")
    scope["modal"] = SimpleNamespace(Workspace=SimpleNamespace(from_context=lambda: workspace))
    monkeypatch.setattr(finance, "official_snapshot", lambda *args, **kwargs: deepcopy(snapshot))
    exec(code, scope)
    job = {
        "schema_version": 1,
        "stage": "pretrain",
        "gross_quota_policy": {"path": "docs/selftrained/gross-quota.json", "sha256": SHA},
    }
    args = (
        "gha-new",
        "selftrained-v2",
        "d" * 40,
        {},
        "e" * 64,
        job,
        "f" * 64,
        {"rates": RATES},
        {},
        runner.sha256(runner.LEDGER),
        policy,
        SHA,
        None,
    )
    return scope["reserve_remote"], args, snapshot, events


@pytest.mark.parametrize(
    "change", ["workspace", "exhausted", "API_failure", "sidecar_race", "ledger_race", "missing_policy"]
)
def test_actual_reserve_body_financial_negatives_before_writes(tmp_path, monkeypatch, runner_time, change):
    function, args, snapshot, events = remote_fixture(tmp_path, monkeypatch)
    before = runner.LEDGER.read_bytes()
    if change == "workspace":
        snapshot["workspace_name"] = "wrong"
    elif change == "exhausted":
        snapshot["summaries"][-1]["metered_cost_usd"] = "36.22341211"
    elif change == "API_failure":
        monkeypatch.setattr(
            finance,
            "official_snapshot",
            lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("synthetic API outage")),
        )
    elif change == "sidecar_race":
        runner.GROSS_QUOTA.write_text("{}")
    elif change == "ledger_race":
        args = args[:9] + ("0" * 64,) + args[10:]
    elif change == "missing_policy":
        args = args[:10] + (None, None, None)
    prior_quota = runner.GROSS_QUOTA.read_bytes() if runner.GROSS_QUOTA.exists() else None
    with pytest.raises((ValueError, RuntimeError, PermissionError)):
        function(*args)
    assert runner.LEDGER.read_bytes() == before
    assert (runner.GROSS_QUOTA.read_bytes() if runner.GROSS_QUOTA.exists() else None) == prior_quota
    assert events == ["reload"]


def test_actual_reserve_body_positive_commits_sidecar_and_preserves_history(tmp_path, monkeypatch, runner_time):
    function, args, snapshot, events = remote_fixture(tmp_path, monkeypatch)
    prior = json.loads(runner.LEDGER.read_text())["reservations"]
    result = function(*args)
    assert events == ["reload", "commit"]
    after = json.loads(runner.LEDGER.read_text())
    assert after["reservations"][:-1] == prior
    assert result["gross_quota_guard"]["authorized_gross_ceiling_usd"] == "36.32341211"
    assert json.loads(runner.GROSS_QUOTA.read_text())["attempts"][0]["run_id"] == "gha-new"


@pytest.mark.parametrize("change", ["drop_month", "shrink_highwater", "snapshot_hash", "last_time"])
def test_stored_highwater_and_snapshot_must_match_exact_prior_ledger(change):
    policy, ledger, snapshot = fixture()
    state, _ = check(ledger=ledger, run_id="gha-prior")
    record(ledger, state)
    if change == "drop_month":
        del state["month_gross_highwater_usd"]["2026-09"]
    elif change == "shrink_highwater":
        state["month_gross_highwater_usd"]["2026-10"] = "6.32341211"
    elif change == "snapshot_hash":
        state["attempts"][0]["billing_snapshot_sha256"] = "0" * 64
    else:
        state["last_observed_at"] = (NOW - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError):
        check(ledger=ledger, state=state)


@pytest.mark.parametrize("change", ["rebase_gross", "raw_baseline", "drop_carry", "drop_carry_pin"])
def test_fixed_authorization_and_actual_evidence_cannot_be_rebased(change):
    policy, ledger, snapshot = fixture()
    if change == "rebase_gross":
        policy["baseline"]["gross_usd"] = "6.41341211"
        policy["baseline"]["cycle_gross_usd"]["2026-10"] = "6.41341211"
    elif change == "raw_baseline":
        policy["baseline"]["raw_sha256"] = "0" * 64
    elif change == "drop_carry":
        policy["carry_in_bounds"] = []
    else:
        policy["carry_in_bounds"][0]["reservation_sha256"] = ""
    with pytest.raises(ValueError):
        check(policy=policy, ledger=ledger, snapshot=snapshot)


def test_financial_policy_metadata_cannot_rerun_same_completed_job(runner_time):
    policy, ledger, snapshot = fixture()
    logical = runner.canonical_sha(
        {"batch_id": "selftrained-v2", "manifest_sha256": "e" * 64, "job": {"schema_version": 1, "stage": "pretrain"}}
    )
    ledger["reservations"][0].update(logical_job_sha256=logical, status="completed")
    with pytest.raises(ValueError, match="already completed"):
        reserve(ledger, policy)


def test_sidecar_stream_size_is_bounded_before_join_or_billing(monkeypatch):
    policy, ledger, snapshot = fixture()
    yielded = []
    billing_calls = []

    def read_file(name):
        if name == "budget.json":
            yield json.dumps(ledger).encode()
        else:
            for count in range(5):
                yielded.append(count)
                yield b"x" * 600

    shared = SimpleNamespace(read_file=read_file)
    workspace = SimpleNamespace(
        billing=SimpleNamespace(rates=lambda: RATES, summary=lambda **kwargs: billing_calls.append(kwargs))
    )
    modal = SimpleNamespace(
        Volume=SimpleNamespace(from_name=lambda *args, **kwargs: shared),
        Workspace=SimpleNamespace(from_context=lambda: workspace),
        exception=SimpleNamespace(NotFoundError=FileNotFoundError),
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setattr(runner, "MAX_LEDGER_BYTES", 1000)
    with pytest.raises(ValueError, match="sidecar exceeds"):
        runner.read_live_ledger(policy, SHA)
    assert yielded == [0, 1] and billing_calls == []
