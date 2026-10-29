"""唯讀核對舊 quota 與 audit ledger；不將歷史總預約冒充未對帳 carry。"""

from decimal import Decimal

from scripts.selftrained import finance as old

POLICY_PATH = "docs/selftrained/v2-gross-quota-20261006.json"


def verify(ledger, quota, policy, policy_sha, now):
    """Validate the old epoch's identities/history without reusing its spending cap.

    舊 validate_policy 僅驗證舊政策本身；本函式不呼叫舊 check_quota，也不
    重設、釋放或消耗舊USD30。舊紀錄沒有已審對帳釋回機制，所以只保留
    sidecar內全部bounds及policy原定carry-in，其他歷史ledger金額只作audit。
    """
    old.validate_policy(policy, policy_sha)
    if (
        quota.get("schema") != old.SCHEMA
        or quota.get("round_id") != policy["round_id"]
        or quota.get("policy_sha256") != policy_sha
        or quota.get("policy_canonical_sha256") != old.canonical_sha(policy)
        or ledger.get("gross_quota_policy_sha256") != policy_sha
    ):
        raise ValueError("Legacy policy, quota sidecar and ledger marker disagree")
    attempts = quota["attempts"]
    if not attempts:
        raise ValueError("Initialized legacy quota must retain its actual attempts")
    records = {item["run_id"]: item for item in ledger["reservations"]}
    if len(records) != len(ledger["reservations"]):
        raise ValueError("Legacy ledger repeats an attempt identity")
    ids = [item["run_id"] for item in attempts]
    carry = policy["carry_in_bounds"]
    carry_ids = {item["run_id"] for item in carry}
    if len(ids) != len(set(ids)) or carry_ids.intersection(ids):
        raise ValueError("Legacy quota/carry identities repeat or overlap")
    scoped = {key for key, value in records.items() if value.get("gross_quota_policy_sha256") == policy_sha}
    if scoped != set(ids):
        raise ValueError("Legacy sidecar does not exactly cover its policy-bound ledger entries")
    selected = []
    for item in [*carry, *attempts]:
        record = records.get(item["run_id"])
        if record is None or old.usd(record["reserved_usd"]) != old.usd(item["reserved_usd"]):
            raise ValueError("Legacy quota/carry bound differs from its actual ledger entry")
        if record.get("status") not in old.TERMINAL:
            raise ValueError("Legacy retained attempt is active or unknown")
        if "reservation_immutable_sha256" in item:
            immutable = {k: v for k, v in record.items() if k not in ("status", "finished_at", "receipt_sha256")}
            if old.canonical_sha(immutable) != item["reservation_immutable_sha256"]:
                raise ValueError("Legacy carry-in immutable reservation fingerprint differs")
        else:
            actual = record["billing_before"]["gross_quota_snapshot"]
            if old.canonical_sha(actual) != item["billing_snapshot_sha256"]:
                raise ValueError("Legacy attempt snapshot differs from sidecar fingerprint")
        selected.append(
            {
                "run_id": item["run_id"],
                "reserved_usd": str(old.usd(item["reserved_usd"])),
                "status": record["status"],
                "receipt_sha256": record.get("receipt_sha256"),
                "scope": "policy-carry-in" if item["run_id"] in carry_ids else "quota-unreleased-bound",
            }
        )
    highwater = {k: old.usd(v) for k, v in policy["baseline"]["cycle_gross_usd"].items()}
    last_observed = old.utc(policy["baseline"]["observed_at"])
    prior_reserved = last_observed
    for item in attempts:
        reserved_at = old.utc(item["reserved_at"])
        actual = records[item["run_id"]]["billing_before"]["gross_quota_snapshot"]
        observed = old.utc(actual["observed_at"])
        if reserved_at < prior_reserved or reserved_at > now or observed < last_observed or observed > reserved_at:
            raise ValueError("Legacy attempt/snapshot chronology is invalid")
        values = old.snapshot_month_values(policy, actual)
        for cycle, value in values.items():
            highwater[cycle] = max(highwater.get(cycle, Decimal(0)), old.usd(value))
        last_observed, prior_reserved = observed, reserved_at
    stored = {k: old.usd(v) for k, v in quota["month_gross_highwater_usd"].items()}
    if stored != highwater or old.utc(quota["last_observed_at"]) != last_observed:
        raise ValueError("Legacy high-water does not match baseline plus all actual snapshots")
    audit_sum = sum((old.usd(value["reserved_usd"]) for value in records.values()), Decimal(0))
    audit_reported = old.usd(ledger["reserved_total_usd"])
    if audit_reported < audit_sum:
        raise ValueError("Legacy reported audit total lost actual reservations")
    return {
        "policy_sha256": policy_sha,
        "policy_canonical_sha256": old.canonical_sha(policy),
        "round_id": policy["round_id"],
        "retained_bounds": selected,
        "carry_usd": str(sum((old.usd(item["reserved_usd"]) for item in selected), Decimal(0))),
        "historical_audit_total_usd": str(audit_reported),
        "historical_audit_excluded_from_carry": True,
        "month_gross_highwater_usd": {k: str(v) for k, v in highwater.items()},
        "last_observed_at": last_observed.isoformat(),
        "termination_evidence": "Status alone is insufficient: independent pinned reconciliation required",
    }
