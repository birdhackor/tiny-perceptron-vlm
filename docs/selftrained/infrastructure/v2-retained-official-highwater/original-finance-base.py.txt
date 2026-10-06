"""Small gross-before-credits quota for one bounded continuation round.

Unreconciled attempts keep their full bound through the round, including failures.
No invoice finality, billing latency, App attribution or credit balance is inferred.
"""

import hashlib
import json
import re
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal

SCHEMA = "selftrained-gross-quota-round-v1"
SOURCE = "Official modal.Workspace.billing.summary"
TERMINAL = {"completed", "failed", "failed-or-cancelled"}
BASELINE_RAW_SHA256 = "3b66653a2963ceeb32c96325780eb35c08a388661a2b90873fa6f44a495382df"
BASELINE_GROSS = Decimal("6.32341211")
BASELINE_TIME = "2026-10-06T09:15:32.796484+00:00"
WORKSPACE = "birdhackor"
CARRY_RUN = "gha-37441599016-1"


def usd(value):
    if isinstance(value, bool):
        raise ValueError("USD cannot be boolean")
    value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise ValueError("USD must be finite and nonnegative")
    return value


def utc(value):
    value = datetime.fromisoformat(value)
    if value.tzinfo is None or value.utcoffset().total_seconds() != 0:
        raise ValueError("Billing times must explicitly identify UTC")
    return value


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def months(first, last):
    result, current = [], utc(first).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    limit = utc(last)
    while current <= limit:
        result.append(current.strftime("%Y-%m"))
        current = current.replace(year=current.year + (current.month == 12), month=current.month % 12 + 1)
    return result


def validate_policy(policy, policy_sha):
    if policy.get("schema") != SCHEMA or not re.fullmatch(r"[a-f0-9]{64}", policy_sha or ""):
        raise ValueError("Gross quota needs its exact committed policy SHA")
    if not isinstance(policy.get("workspace_name"), str) or not policy["workspace_name"]:
        raise ValueError("Gross quota requires the authenticated public workspace name")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", policy.get("round_id", "")):
        raise ValueError("Gross quota requires a fixed round ID")
    baseline = policy["baseline"]
    baseline_time = utc(baseline["observed_at"])
    cycles = baseline["cycle_gross_usd"]
    if (
        policy["workspace_name"] != WORKSPACE
        or baseline_time != utc(BASELINE_TIME)
        or usd(baseline["gross_usd"]) != BASELINE_GROSS
        or baseline.get("raw_sha256") != BASELINE_RAW_SHA256
        or {key: usd(value) for key, value in cycles.items()} != {"2026-09": Decimal(0), "2026-10": BASELINE_GROSS}
    ):
        raise ValueError("This authorized round must retain the exact actual baseline and workspace")
    if not cycles or sorted(cycles) != months(min(cycles) + "-01T00:00:00+00:00", baseline_time.isoformat()):
        raise ValueError("Baseline monthly gross coverage must be complete")
    if sum((usd(value) for value in cycles.values()), Decimal(0)) != usd(baseline["gross_usd"]):
        raise ValueError("Baseline gross must equal its recorded monthly gross values")
    if not re.fullmatch(r"[a-f0-9]{64}", baseline.get("raw_sha256", "")):
        raise ValueError("Baseline must retain its original official raw-byte SHA")
    if usd(policy["additional_quota_usd"]) != Decimal("30"):
        raise ValueError("This continuation round is authorized for exactly additional USD30")
    if type(policy.get("snapshot_max_age_seconds")) is not int or not 1 <= policy["snapshot_max_age_seconds"] <= 120:
        raise ValueError("Gross snapshots require a bounded observation freshness check")
    seen = set()
    if [(item["run_id"], usd(item["reserved_usd"])) for item in policy["carry_in_bounds"]] != [
        (CARRY_RUN, Decimal("0.54"))
    ]:
        raise ValueError("This round must retain the known pre-baseline active attempt's USD0.54 bound")
    for item in policy["carry_in_bounds"]:
        if item["run_id"] in seen or usd(item["reserved_usd"]) == 0:
            raise ValueError("Carry-in needs unique actual attempts and positive bounds")
        seen.add(item["run_id"])
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("reservation_sha256", "")):
            raise ValueError("Carry-in reservation evidence must be pinned before enabling quota")
        if not re.fullmatch(r"[a-f0-9]{64}", item.get("reservation_immutable_sha256", "")):
            raise ValueError("Carry-in immutable reservation identity must be pinned")
    return policy


def official_snapshot(workspace, policy, policy_sha, now=None):
    """Public summary only: no report windows, private APIs, credits or App compute."""
    validate_policy(policy, policy_sha)
    started = now or datetime.now(UTC)
    cycles = months(min(policy["baseline"]["cycle_gross_usd"]) + "-01T00:00:00+00:00", started.isoformat())
    summaries = []
    for cycle in cycles:
        value = workspace.billing.summary(cycle=cycle)
        summaries.append(
            {
                "cycle": cycle,
                "start": value.start.isoformat(),
                "end": value.end.isoformat(),
                "metered_cost_usd": str(usd(value.metered_cost)),
            }
        )
    # summary hydrates the public Workspace before reading its public name.
    return {
        "source": SOURCE,
        "workspace_name": workspace.name,
        "observed_at": (now or datetime.now(UTC)).isoformat(),
        "summaries": summaries,
    }


def check_quota(policy, policy_sha, state, snapshot, ledger, proposed_usd, run_id, now=None):
    """Pure guard. The caller writes state only after all checks pass."""
    validate_policy(policy, policy_sha)
    now = now or datetime.now(UTC)
    observed = utc(snapshot["observed_at"])
    if (
        observed > now
        or (now - observed).total_seconds() > policy["snapshot_max_age_seconds"]
        or observed < utc(policy["baseline"]["observed_at"])
    ):
        raise ValueError("Gross snapshot is stale, future-dated or precedes the baseline")
    if snapshot.get("source") != SOURCE or snapshot.get("workspace_name") != policy["workspace_name"]:
        raise ValueError("Official snapshot must identify the same authenticated workspace")
    expected_cycles = months(min(policy["baseline"]["cycle_gross_usd"]) + "-01T00:00:00+00:00", observed.isoformat())
    summaries = snapshot["summaries"]
    if [item["cycle"] for item in summaries] != expected_cycles:
        raise ValueError("Official monthly gross cycles are missing, duplicate or reordered")
    current = {}
    for item in summaries:
        start = utc(item["start"])
        end = utc(item["end"])
        if start.isoformat() != item["cycle"] + "-01T00:00:00+00:00" or end != start.replace(
            year=start.year + (start.month == 12), month=start.month % 12 + 1
        ):
            raise ValueError("Official summary has the wrong UTC calendar-month boundaries")
        current[item["cycle"]] = str(usd(item["metered_cost_usd"]))
    if state is not None:
        if (
            state.get("schema") != SCHEMA
            or state.get("round_id") != policy["round_id"]
            or state.get("policy_sha256") != policy_sha
            or state.get("policy_canonical_sha256") != canonical_sha(policy)
        ):
            raise ValueError("Gross quota round/policy cannot be reset or silently replaced")
        if observed < utc(state["last_observed_at"]):
            raise ValueError("Gross quota observations must not move backwards")
    previous = policy["baseline"]["cycle_gross_usd"] if state is None else state["month_gross_highwater_usd"]
    if state is not None:
        if utc(state["last_observed_at"]) < utc(policy["baseline"]["observed_at"]) or sorted(previous) != months(
            min(policy["baseline"]["cycle_gross_usd"]) + "-01T00:00:00+00:00", state["last_observed_at"]
        ):
            raise ValueError("Stored gross high-water coverage must retain every baseline-to-observation month")
    if any(usd(current[cycle]) < usd(value) for cycle, value in policy["baseline"]["cycle_gross_usd"].items()):
        raise ValueError("Official gross is below its per-month pinned baseline")
    if any(cycle not in current or usd(current[cycle]) < usd(value) for cycle, value in previous.items()):
        raise ValueError("Official gross regressed; reconcile without silently refunding quota")
    attempts = [] if state is None else state["attempts"]
    if state is not None and not attempts:
        raise ValueError("An initialized quota sidecar must retain its actual attempts")
    prior_time = utc(policy["baseline"]["observed_at"])
    for item in attempts:
        reserved_at = utc(item["reserved_at"])
        if reserved_at < prior_time or reserved_at > now:
            raise ValueError("Attempt reservation times must be ordered within this round")
        prior_time = reserved_at
    ids = [item["run_id"] for item in attempts]
    if len(ids) != len(set(ids)) or run_id in ids:
        raise ValueError("Gross quota attempts require fresh unique run IDs")
    records = {item["run_id"]: item for item in ledger["reservations"]}
    if len(records) != len(ledger["reservations"]):
        raise ValueError("Shared ledger repeats an attempt ID")
    if run_id in records:
        raise ValueError("New quota attempt already exists in the audit ledger")
    carry_ids = {item["run_id"] for item in policy["carry_in_bounds"]}
    recorded_ids = {name for name, item in records.items() if item.get("gross_quota_policy_sha256") == policy_sha}
    if recorded_ids != set(ids) or carry_ids.intersection(ids):
        raise ValueError(
            "Quota sidecar attempts must exactly match this policy's audit ledger entries without carry-in overlap"
        )
    for item in [*policy["carry_in_bounds"], *attempts]:
        record = records.get(item["run_id"])
        if record is None or usd(record["reserved_usd"]) != usd(item["reserved_usd"]):
            raise ValueError("Gross quota bound differs from its actual legacy reservation")
        if record.get("status") not in TERMINAL:
            raise RuntimeError("Sole-paid/no-overlap guard: prior bound attempt is not terminal")
        if "reservation_immutable_sha256" in item:
            immutable = {
                key: value for key, value in record.items() if key not in ("status", "finished_at", "receipt_sha256")
            }
            if canonical_sha(immutable) != item["reservation_immutable_sha256"]:
                raise ValueError("Carry-in differs from its pinned immutable actual reservation")
        elif record.get("gross_quota_policy_sha256") != policy_sha:
            raise ValueError("This-round attempt is missing its exact policy binding in the audit ledger")
        else:
            actual_snapshot = record["billing_before"]["gross_quota_snapshot"]
            if canonical_sha(actual_snapshot) != item["billing_snapshot_sha256"]:
                raise ValueError("Attempt billing snapshot differs from its actual audit ledger binding")
    if state is not None:
        last_snapshot = records[attempts[-1]["run_id"]]["billing_before"]["gross_quota_snapshot"]
        recorded_highwater = {item["cycle"]: str(usd(item["metered_cost_usd"])) for item in last_snapshot["summaries"]}
        if utc(last_snapshot["observed_at"]) != utc(state["last_observed_at"]) or recorded_highwater != previous:
            raise ValueError("Sidecar gross high-water differs from its last exact ledger snapshot")
    gross = sum((usd(value) for value in current.values()), Decimal(0))
    increment = gross - usd(policy["baseline"]["gross_usd"])
    pending = sum((usd(item["reserved_usd"]) for item in attempts), Decimal(0))
    carry = sum((usd(item["reserved_usd"]) for item in policy["carry_in_bounds"]), Decimal(0))
    proposed = usd(proposed_usd)
    guarded = increment + pending + carry + proposed
    if guarded > usd(policy["additional_quota_usd"]):
        raise RuntimeError(
            f"Gross quota exhausted: actualincrement {increment} + unreconciled {pending} + carry-in {carry} + proposed {proposed} exceeds USD30"
        )
    result = {
        "schema": SCHEMA,
        "policy_sha256": policy_sha,
        "policy_canonical_sha256": canonical_sha(policy),
        "round_id": policy["round_id"],
        "last_observed_at": observed.isoformat(),
        "month_gross_highwater_usd": current,
        "attempts": deepcopy(attempts),
    }
    result["attempts"].append(
        {
            "run_id": run_id,
            "reserved_usd": str(proposed),
            "reserved_at": now.isoformat(),
            "billing_snapshot_sha256": canonical_sha(snapshot),
        }
    )
    return result, {
        "official_current_gross_usd": str(gross),
        "official_baseline_gross_usd": policy["baseline"]["gross_usd"],
        "authorized_gross_ceiling_usd": str(usd(policy["baseline"]["gross_usd"]) + usd(policy["additional_quota_usd"])),
        "actual_gross_increment_usd": str(increment),
        "unreconciled_round_bounds_usd": str(pending),
        "carry_in_bounds_usd": str(carry),
        "proposed_bound_usd": str(proposed),
        "guarded_increment_usd": str(guarded),
        "remaining_guarded_quota_usd": str(usd(policy["additional_quota_usd"]) - guarded),
        "settlement": "No automatic release; all this-round bounds retained until a separately audited reconciliation",
        "limitations": "Includes workspace other projects in observed gross; arbitrary unobserved charges outside registered bounds and future storage cannot be bounded by this snapshot; freshness is not billing finality or a maximum billing lag",
    }
