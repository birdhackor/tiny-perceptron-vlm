"""獨立USD60 round：先持久預約、保留官方gross及全部未釋放bounds。

純client P1不會派任何Modal container。此模組為日後已審GPU runner保留
reserve/finish介面；它不替任何GPU形狀、映像或preemption提供費用證明。
"""

import base64
import hashlib
import json
import re
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from urllib.parse import quote

POLICY = {
    "schema": "omni-revision-gross-round-v2",
    "round_id": "omni-20261029",
    "workspace": "birdhackor",
    "environment": "main",
    "additional_gross_cap_usd": "60.00",
    "snapshot_max_age_seconds": 120,
    "authority": "User approved v3 including the new USD60 round: 可以，GO",
    "guard": "retained_official_increment + all_new_unreleased_bounds + exact_legacy_carry + proposed <= 60",
    "second_guard": "sum_of_all_new_bounds <= 60; no automatic release",
    "legacy_policy": "Read-only; historical audit totals are not unreconciled carry",
}
REPOSITORY = "birdhackor/tiny-perceptron-vlm"
BUDGET_BRANCH = "budget/omni-20261029"
BUDGET_PATH = "omni-budget.json"
SOURCE = "Official modal.Workspace.billing.summary"
TERMINAL = {"completed", "failed", "failed-or-cancelled"}
COMPONENTS = {"compute", "startup_idle", "build", "egress", "storage_cleanup"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def usd(value):
    if isinstance(value, bool):
        raise ValueError("USD cannot be boolean")
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("USD must be finite and nonnegative")
    return number


def utc(value):
    time = datetime.fromisoformat(value)
    if time.tzinfo is None or time.utcoffset().total_seconds() != 0:
        raise ValueError("Time must explicitly identify UTC")
    return time


def cycle_range(first, last):
    current = utc(first).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    result = []
    while current <= utc(last):
        result.append(current.strftime("%Y-%m"))
        current = current.replace(year=current.year + (current.month == 12), month=current.month % 12 + 1)
    return result


def snapshot_values(snapshot, baseline_time, now, fresh=True, first_cycle=None):
    if snapshot.get("source") != SOURCE or snapshot.get("workspace") != POLICY["workspace"]:
        raise ValueError("Need official billing from the authorized workspace")
    observed = utc(snapshot["observed_at"])
    if observed > now or (fresh and (now - observed).total_seconds() > POLICY["snapshot_max_age_seconds"]):
        raise ValueError("Official snapshot is stale or future-dated")
    if observed < utc(baseline_time):
        raise ValueError("Official snapshot precedes the new-round baseline")
    rows = snapshot["summaries"]
    if not rows:
        raise ValueError("Official monthly summaries cannot be empty")
    first_cycle = first_cycle or rows[0]["cycle"]
    if [row["cycle"] for row in rows] != cycle_range(first_cycle + "-01T00:00:00+00:00", observed.isoformat()):
        raise ValueError("Monthly gross coverage must be complete, unique and ordered")
    values = {}
    for row in rows:
        start, end = utc(row["start"]), utc(row["end"])
        if start.isoformat() != row["cycle"] + "-01T00:00:00+00:00" or end != start.replace(
            year=start.year + (start.month == 12), month=start.month % 12 + 1
        ):
            raise ValueError("Official calendar-month boundaries are invalid")
        values[row["cycle"]] = usd(row["metered_cost_usd"])
    return values


def legacy_carry(inventory):
    report, reconciliation = inventory["legacy_verification"], inventory["reconciliation"]
    if not report.get("historical_audit_excluded_from_carry"):
        raise ValueError("Historical audit totals must not be used as legacy carry")
    if reconciliation.get("schema") != "omni-legacy-reconciliation-v1":
        raise ValueError("Independent pinned legacy reconciliation is required")
    if not re.fullmatch(r"/root/[A-Za-z0-9_/.-]+", reconciliation.get("reviewer_task", "")):
        raise ValueError("Legacy reconciliation must identify its independent reviewer")
    if reconciliation.get("legacy_verification_sha256") != digest(report):
        raise ValueError("Reconciliation does not match the actual validated old quota")
    if reconciliation.get("existing_volumes_sha256") != digest(inventory["existing_volumes"]):
        raise ValueError("Existing persistent storage must match independent reconciliation")
    for key in ("ledger_sha256", "quota_sha256", "legacy_policy_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", inventory.get(key, "")) or reconciliation.get(key) != inventory[key]:
            raise ValueError("Legacy raw/policy fingerprints differ from independent reconciliation")
    selected = report["retained_bounds"]
    expected = {item["run_id"]: usd(item["reserved_usd"]) for item in selected}
    if len(expected) != len(selected) or any(item["status"] not in TERMINAL for item in selected):
        raise ValueError("Legacy carry identities are duplicate, active or unknown")
    verified = reconciliation["retained_bounds"]
    actual = {item["run_id"]: usd(item["reserved_usd"]) for item in verified}
    if len(actual) != len(verified) or actual != expected:
        raise ValueError("Reconciliation must retain every exact old quota/carry bound")
    for item in verified:
        if not item.get("stop_evidence"):
            raise ValueError("Each retained old attempt needs independent termination evidence")
        for evidence in item["stop_evidence"]:
            if (
                not evidence.get("path", "").startswith("docs/")
                or ".." in evidence["path"].split("/")
                or not re.fullmatch(r"[a-f0-9]{64}", evidence.get("sha256", ""))
            ):
                raise ValueError("Stopping evidence must identify committed documentary bytes")
    amount = sum(expected.values(), Decimal(0))
    if usd(report["carry_usd"]) != amount:
        raise ValueError("Legacy carry total differs from the validated exact bound list")
    return amount


def validate_inventory(inventory, now):
    if inventory.get("workspace") != POLICY["workspace"] or inventory.get("environment") != POLICY["environment"]:
        raise ValueError("Inventory belongs to a different workspace/environment")
    if inventory.get("active_apps") or inventory.get("unknown_attempts") or inventory.get("errors"):
        raise ValueError("Legacy active/unknown resources must be reconciled before any paid work")
    observed = utc(inventory["observed_at"])
    if observed > now or (now - observed).total_seconds() > 120:
        raise ValueError("Legacy inventory is stale or future-dated")
    return legacy_carry(inventory)


def new_state(snapshot, inventory, revision, now=None):
    now = now or datetime.now(UTC)
    carry = validate_inventory(inventory, now)
    old_highwater = inventory["legacy_verification"]["month_gross_highwater_usd"]
    values = snapshot_values(snapshot, snapshot["observed_at"], now, first_cycle=min(old_highwater))
    if any(values.get(cycle, Decimal(0)) < usd(value) for cycle, value in old_highwater.items()):
        raise ValueError("New official baseline is below retained legacy monthly high-water")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Initialization needs an immutable code revision")
    if carry > usd(POLICY["additional_gross_cap_usd"]):
        raise ValueError("Legacy unreconciled carry already exceeds the new conservative cap")
    origin = {
        "baseline": snapshot,
        "legacy_inventory": inventory,
        "legacy_carry_usd": str(carry),
        "initialization_revision": revision,
        "policy_sha256": digest(POLICY),
    }
    return {
        "schema": POLICY["schema"],
        "policy": deepcopy(POLICY),
        "origin": origin,
        "origin_sha256": digest(origin),
        "month_gross_highwater_usd": {k: str(v) for k, v in values.items()},
        "last_observed_at": snapshot["observed_at"],
        "attempts": [],
        "reserved_total_usd": "0",
    }


def reservation_identity(attempt):
    return {k: v for k, v in attempt.items() if k not in {"status", "receipt", "reservation_sha256"}}


def validate_state(state, now):
    if state.get("schema") != POLICY["schema"] or state.get("policy") != POLICY:
        raise ValueError("New-round policy cannot be replaced or reset")
    origin = state["origin"]
    if state.get("origin_sha256") != digest(origin) or origin.get("policy_sha256") != digest(POLICY):
        raise ValueError("Initialization baseline/policy/inventory identity changed")
    baseline = origin["baseline"]
    highwater = snapshot_values(baseline, baseline["observed_at"], now, fresh=False)
    first_cycle = baseline["summaries"][0]["cycle"]
    baseline_total = sum(highwater.values(), Decimal(0))
    carry = legacy_carry(origin["legacy_inventory"])
    if carry != usd(origin["legacy_carry_usd"]):
        raise ValueError("Legacy carry no longer matches the original exact reconciliation")
    total, ids = Decimal(0), set()
    last_observed = utc(baseline["observed_at"])
    prior_reserved = last_observed
    for attempt in state["attempts"]:
        if attempt["attempt_id"] in ids:
            raise ValueError("Stored history repeats an attempt identity")
        ids.add(attempt["attempt_id"])
        if attempt.get("reservation_sha256") != digest(reservation_identity(attempt)):
            raise ValueError("Stored reservation identity/bound/snapshot was changed")
        if set(attempt["components_usd"]) != COMPONENTS:
            raise ValueError("Stored reservation lost a cost boundary")
        amount = sum((usd(x) for x in attempt["components_usd"].values()), Decimal(0))
        if amount <= 0 or amount != usd(attempt["reserved_usd"]):
            raise ValueError("Stored reservation amount differs from its component bounds")
        when = utc(attempt["reserved_at"])
        observed = utc(attempt["billing_snapshot"]["observed_at"])
        if when < prior_reserved or when > now or observed < last_observed or observed > when:
            raise ValueError("Stored reservation/snapshot chronology is invalid")
        values = snapshot_values(attempt["billing_snapshot"], baseline["observed_at"], when, first_cycle=first_cycle)
        for cycle, value in values.items():
            highwater[cycle] = max(highwater.get(cycle, Decimal(0)), value)
        increment = sum(highwater.values(), Decimal(0)) - baseline_total
        if increment < 0 or increment + total + carry + amount > usd(POLICY["additional_gross_cap_usd"]):
            raise ValueError("Stored history violates the retained official gross guard")
        expected_guard = {
            "retained_official_increment_usd": str(increment),
            "prior_new_bounds_usd": str(total),
            "exact_legacy_carry_usd": str(carry),
            "proposed_bound_usd": str(amount),
            "guarded_increment_usd": str(increment + total + carry + amount),
        }
        if attempt["guard"] != expected_guard:
            raise ValueError("Stored guard arithmetic differs from its complete actual history")
        total += amount
        if total > usd(POLICY["additional_gross_cap_usd"]):
            raise ValueError("Stored new bounds exceed USD60")
        last_observed, prior_reserved = observed, when
        if attempt["status"] not in {"reserved", "unresolved", "completed", "failed-stopped"}:
            raise ValueError("Unknown new attempt state")
        if attempt["status"] in {"completed", "failed-stopped"} and not resources_verified(attempt["receipt"]):
            raise ValueError("Terminal state lacks verified compute/resource cleanup")
    if (
        {k: usd(v) for k, v in state["month_gross_highwater_usd"].items()} != highwater
        or utc(state["last_observed_at"]) != last_observed
        or usd(state["reserved_total_usd"]) != total
    ):
        raise ValueError("Stored high-water/total does not match baseline plus complete reservation history")
    return highwater, total, carry


def reserve(state, snapshot, inventory, attempt_id, components, revision, purpose, now=None):
    """Return a pure guarded transition; persist it by CAS before any paid call."""
    now = now or datetime.now(UTC)
    highwater, previous, carry = validate_state(state, now)
    current_carry = validate_inventory(inventory, now)
    initial = state["origin"]["legacy_inventory"]
    keys = ("ledger_sha256", "quota_sha256", "legacy_policy_sha256")
    if current_carry != carry or any(inventory[k] != initial[k] for k in keys):
        raise ValueError("Legacy raw files/policy/carry changed; independently reconcile before continuing")
    if inventory["reconciliation"] != initial["reconciliation"]:
        raise ValueError("Pinned independent reconciliation cannot be replaced")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", attempt_id) or ".." in attempt_id:
        raise ValueError("Attempt identity is unsafe")
    if not re.fullmatch(r"[a-f0-9]{40}", revision) or not purpose:
        raise ValueError("Reservation needs the actual code revision and purpose")
    if any(x["attempt_id"] == attempt_id for x in state["attempts"]):
        raise ValueError("Attempt identities cannot be reused, even after failure")
    if any(x["status"] not in {"completed", "failed-stopped"} for x in state["attempts"]):
        raise ValueError("A prior reservation is still running or unresolved")
    if set(components) != COMPONENTS:
        raise ValueError("Every paid boundary must have an explicit cost component")
    amount = sum((usd(x) for x in components.values()), Decimal(0))
    if amount <= 0:
        raise ValueError("A paid attempt needs a positive reservation")
    cap = usd(POLICY["additional_gross_cap_usd"])
    if previous + amount > cap:
        raise ValueError("New-round conservative reservations would exceed USD60")
    baseline = state["origin"]["baseline"]
    values = snapshot_values(snapshot, baseline["observed_at"], now, first_cycle=baseline["summaries"][0]["cycle"])
    if utc(snapshot["observed_at"]) < utc(state["last_observed_at"]):
        raise ValueError("Snapshot observation time moved backwards")
    retained = {k: max(v, highwater.get(k, Decimal(0))) for k, v in values.items()}
    baseline_total = sum((usd(row["metered_cost_usd"]) for row in baseline["summaries"]), Decimal(0))
    increment = sum(retained.values(), Decimal(0)) - baseline_total
    if any(values.get(k, Decimal(0)) < v for k, v in highwater.items()):
        raise ValueError("Official gross moved below its reconstructed monthly high-water")
    if increment > previous + carry:
        raise ValueError("Unexplained workspace gross increment; stop for accounting")
    guarded = increment + previous + carry + amount
    if guarded > cap:
        raise ValueError("Retained official increment + all new bounds + exact legacy carry + proposed exceeds USD60")
    following = deepcopy(state)
    attempt = {
        "attempt_id": attempt_id,
        "reserved_usd": str(amount),
        "components_usd": {k: str(usd(v)) for k, v in components.items()},
        "reserved_at": now.isoformat(),
        "revision": revision,
        "purpose": purpose,
        "billing_snapshot": snapshot,
        "legacy_inventory_sha256": digest(inventory),
        "guard": {
            "retained_official_increment_usd": str(increment),
            "prior_new_bounds_usd": str(previous),
            "exact_legacy_carry_usd": str(carry),
            "proposed_bound_usd": str(amount),
            "guarded_increment_usd": str(guarded),
        },
        "status": "reserved",
    }
    attempt["reservation_sha256"] = digest(reservation_identity(attempt))
    following["attempts"].append(attempt)
    following.update(
        reserved_total_usd=str(previous + amount),
        month_gross_highwater_usd={k: str(v) for k, v in retained.items()},
        last_observed_at=snapshot["observed_at"],
    )
    return following


def resources_verified(receipt):
    """P4 must define and independently verify every actual owned resource."""
    checks = receipt.get("cleanup_checks")
    return (
        receipt.get("compute_stopped") is True
        and receipt.get("resources_verified") is True
        and isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values())
    )


def finish(state, attempt_id, receipt):
    """No refund: unknown resource/cleanup states remain unresolved."""
    following = deepcopy(state)
    matches = [x for x in following["attempts"] if x["attempt_id"] == attempt_id]
    if len(matches) != 1 or matches[0]["status"] != "reserved":
        raise ValueError("Only the exact reserved attempt can be finished once")
    verified = resources_verified(receipt)
    matches[0]["status"] = (
        "completed" if verified and receipt.get("success") is True else "failed-stopped" if verified else "unresolved"
    )
    matches[0]["receipt"] = receipt
    return following


class GitHubLedger:
    """Only GH API calls; contents PUT uses the previously observed blob SHA."""

    def __init__(self, repository=REPOSITORY):
        if repository != REPOSITORY:
            raise ValueError("Budget ledger must be in the authorized course repository")
        self.repository = repository

    def api(self, endpoint, payload=None, allow_missing=False):
        command = ["gh", "api", f"repos/{self.repository}/{endpoint}"]
        if payload is not None:
            command += ["--method", "POST" if endpoint == "git/refs" else "PUT", "--input", "-"]
        completed = subprocess.run(
            command, input=canonical(payload) if payload is not None else None, capture_output=True
        )
        if completed.returncode:
            if allow_missing and b"HTTP 404" in completed.stderr:
                return None
            raise RuntimeError("GitHub ledger API rejected the transaction; no paid work may start")
        return json.loads(completed.stdout)

    def read(self):
        value = self.api(f"contents/{BUDGET_PATH}?ref={quote(BUDGET_BRANCH, safe='')}", allow_missing=True)
        if value is None:
            return None, None
        raw = base64.b64decode(value["content"], validate=False)
        if len(raw) > 1024 * 1024:
            raise ValueError("Budget ledger exceeds its 1MiB bound")
        return json.loads(raw), value["sha"]

    def initialize(self, state, revision):
        if self.api(f"git/ref/heads/{BUDGET_BRANCH}", allow_missing=True) is not None:
            raise ValueError("Round branch already exists; initialization must never reset it")
        self.api("git/refs", {"ref": "refs/heads/" + BUDGET_BRANCH, "sha": revision})
        return self.write(state, None)

    def write(self, state, observed_sha):
        payload = {
            "message": f"[skip ci] Persist {POLICY['round_id']} budget transaction",
            "branch": BUDGET_BRANCH,
            "content": base64.b64encode(canonical(state) + b"\n").decode(),
        }
        if observed_sha is not None:
            payload["sha"] = observed_sha
        response = self.api(f"contents/{BUDGET_PATH}", payload)
        return {"commit": response["commit"]["sha"], "blob": response["content"]["sha"]}
