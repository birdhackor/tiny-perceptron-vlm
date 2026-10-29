"""僅處理已存在的第一次探測；保留0.40與原receipt，無派工／退款。"""

import argparse
import asyncio
import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from scripts.omni_revision import attempt_inspection, preflight
from scripts.omni_revision.finance import GitHubLedger, digest, reservation_identity, utc, validate_state

ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = "probe-40957487215-1"
ORIGINAL_REVISION = "90f9a75611e91bf927c9116b94e025fe8d854947"
ORIGINAL_RECEIPT_SHA256 = "9732a8c09e2bbd1f62821943478df9a348600ef4b4d59b873a5b82cb721badd6"
RESERVATION_SHA256 = "195207edf17950e0e4cac792966d72b0a089aece160748265310205b58b4f377"
APP_ID = "ap-S6YMhO8bLq0D4qVHhFKFDM"
AUDIT = "docs/course-revision-20261029-omni/infrastructure/execution-probe-reconciliation-audit-v1.json"
SOURCES = (
    "scripts/omni_revision/reconcile_failed_probe.py",
    "scripts/omni_revision/attempt_inspection.py",
    "scripts/omni_revision/finance.py",
    "scripts/omni_revision/preflight.py",
    ".github/workflows/omni-execution.yml",
    "tests/test_omni_probe_reconciliation.py",
    "docs/course-revision-20261029-omni/infrastructure/failed-probe-attempt-v1.json",
)


def resolve(state, inspection, now=None):
    """原RPC拒絕的來源推導由獨立audit限定本次；不接受一般InvalidError。"""
    now = now or datetime.now(UTC)
    if len(state["attempts"]) != 1:
        raise ValueError("只適用本round第一次且唯一失敗探測")
    original = state["attempts"][0]
    if (
        original["attempt_id"] != ATTEMPT
        or original["status"] != "unresolved"
        or original["revision"] != ORIGINAL_REVISION
        or original["reserved_usd"] != "0.40"
        or original["reservation_sha256"] != RESERVATION_SHA256
        or digest(reservation_identity(original)) != original["reservation_sha256"]
        or digest(original["receipt"]) != ORIGINAL_RECEIPT_SHA256
    ):
        raise ValueError("原attempt／金額／revision／receipt身份不符")
    observed = utc(inspection["observed_at"])
    if not 0 <= (now - observed).total_seconds() <= 120:
        raise ValueError("必須重新取得120秒內的官方terminal查核")
    info = inspection["app_info"]["info"]
    names = json.loads(original["purpose"])["ownership"]
    if (
        inspection.get("workspace") != "birdhackor"
        or inspection.get("environment") != "main"
        or inspection.get("attempt_id") != ATTEMPT
        or inspection.get("reservation_sha256") != RESERVATION_SHA256
        or inspection.get("ownership") != names
        or info.get("app_id") != APP_ID
        or info.get("description") != names["app_name"]
        or info.get("environment_name") != "main"
        or inspection.get("remote_mutation") is not False
        or inspection.get("all_sandboxes_include_finished") != []
        or inspection.get("rpc_allowlist") != ["WorkspaceNameLookup", "AppGetInfo", "SandboxList"]
    ):
        raise ValueError("官方exact App／完整finished-inclusive history不符")
    lifecycle = info["lifecycle"]
    stopped = datetime.fromtimestamp(lifecycle["stopped_at"], UTC)
    if (
        lifecycle.get("app_state") != "APP_STATE_STOPPED"
        or not utc(original["receipt"]["paid_call_at"]) <= stopped <= observed
    ):
        raise ValueError("本次App尚未獲官方terminal lifecycle證明")
    following = deepcopy(state)
    following["attempts"][0].update(
        status="failed-stopped",
        receipt={
            "success": False,
            "compute_stopped": True,
            "resources_verified": True,
            "cleanup_checks": {
                "exact_original_rejected_attempt": True,
                "official_terminal_app_lifecycle": True,
                "complete_finished_inclusive_history_has_no_instances": True,
                "no_bound_released": True,
            },
            "original_receipt": deepcopy(original["receipt"]),
            "original_receipt_sha256": ORIGINAL_RECEIPT_SHA256,
            "original_terminal_commit": "9b8c90a4d1e7256c1527bb4d53085b8063ec41d5",
            "resolved_at": now.isoformat(),
            "official_inspection": inspection,
            "source_inference": "Only this exact signed source path: InvalidError before any Sandbox ID came from the single SandboxCreate RPC. Original wire status/message was not saved; this is a source-derived rejected-request finding, not an observed original wire status. No GPU success is claimed.",
        },
    )
    assert following["reserved_total_usd"] == state["reserved_total_usd"]
    return following


def checked_audit():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    audit = json.loads((ROOT / AUDIT).read_bytes())
    if (
        audit.get("verdict") != "pass"
        or audit.get("reviewer_task") != "/root/omni_execution_probe_review"
        or audit.get("only_attempt_id") != ATTEMPT
    ):
        raise ValueError("需要本次one-off reconciliation的獨立稽核")
    if set(audit["files"]) != set(SOURCES):
        raise ValueError("稽核source scope不完整")
    for name in (*SOURCES, AUDIT):
        raw = (ROOT / name).read_bytes()
        if subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT) != raw:
            raise ValueError("source／audit必須等於已提交revision")
        if name != AUDIT and hashlib.sha256(raw).hexdigest() != audit["files"][name]:
            raise ValueError("one-off來源偏離獨立稽核")
    return revision, audit


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision, audit = checked_audit()
    store = GitHubLedger()
    state, blob = store.read()
    validate_state(state, datetime.now(UTC))
    if len(state["attempts"]) != 1 or state["attempts"][0]["attempt_id"] != ATTEMPT:
        raise ValueError("只能查核已存在的唯一失敗attempt")
    inspection = asyncio.run(attempt_inspection.inspect(state["attempts"][0]))
    following = resolve(state, inspection)
    following["attempts"][0]["receipt"]["reconciliation_authority"] = {
        "revision": revision,
        "audit_path": AUDIT,
        "audit_sha256": hashlib.sha256((ROOT / AUDIT).read_bytes()).hexdigest(),
    }
    validate_state(following, datetime.now(UTC))
    preflight.write_json(
        args.output / "reconciliation-before-write.json",
        {"official_inspection": inspection, "following_attempt": following["attempts"][0]},
    )
    transaction = store.write(following, blob)  # 唯一寫入；不退款、不重設原額、不自動CAS重試。
    preflight.write_json(
        args.output / "reconciliation-transaction.json",
        {"transaction": transaction, "attempt": following["attempts"][0]},
    )


if __name__ == "__main__":
    main()
