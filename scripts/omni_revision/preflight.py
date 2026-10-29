"""P1只做client唯讀inventory及GitHub獨立round初始化。

本檔沒有Modal App、Image、Secret、CPU/GPU dispatch、named Volume建立或
寫入。GitHub transport probe由獨立腳本／workflow step執行，不需要Modal。
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.omni_revision import legacy
from scripts.omni_revision.finance import POLICY, SOURCE, TERMINAL, GitHubLedger, cycle_range, digest, new_state

ROOT = Path(__file__).resolve().parents[2]
MAX_LEGACY_BYTES = 4 * 1024 * 1024
DEFAULT_RECONCILIATION = "docs/course-revision-20261029-omni/legacy-reconciliation.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def app_rows():
    completed = subprocess.run(
        [sys.executable, "-m", "modal", "app", "list", "--env", POLICY["environment"], "--json"],
        check=True,
        capture_output=True,
        timeout=40,
    )
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list):
        raise ValueError("Modal app list did not return a JSON list")
    return rows


def stopped_app(row):
    return str(row.get("state", "")).lower() == "stopped" and str(row.get("tasks", "")) == "0"


def read_bounded(volume, name):
    chunks, count = [], 0
    for chunk in volume.read_file(name):
        count += len(chunk)
        if count > MAX_LEGACY_BYTES:
            raise ValueError("Legacy inventory file exceeds 4MiB")
        chunks.append(chunk)
    return b"".join(chunks)


def inventory(modal, output, reconciliation=None):
    """Preserve raw bytes even when quota validation or reconciliation is blocked."""
    volume = modal.Volume.from_name("tiny-perceptron-course", create_if_missing=False)
    raw_ledger = read_bounded(volume, "budget.json")
    raw_quota = read_bounded(volume, "gross-quota.json")
    raw_policy = (ROOT / legacy.POLICY_PATH).read_bytes()
    output.mkdir(parents=True, exist_ok=True)
    (output / "legacy-budget.json").write_bytes(raw_ledger)
    (output / "legacy-gross-quota.json").write_bytes(raw_quota)
    (output / "legacy-policy.json").write_bytes(raw_policy)
    ledger, quota, policy = map(json.loads, (raw_ledger, raw_quota, raw_policy))
    rows = app_rows()
    unknown = [
        {
            "run_id": item.get("run_id"),
            "status": item.get("status", "unknown"),
            "reserved_usd": item.get("reserved_usd"),
        }
        for item in ledger["reservations"]
        if item.get("status") not in TERMINAL
    ]
    volumes = []
    for item in modal.Volume.objects.list(environment_name=POLICY["environment"]):
        info = item.info()
        volumes.append({"name": info.name, "object_id": item.object_id, "created_at": info.created_at.isoformat()})
    volumes.sort(key=lambda item: (item["name"] or "", item["object_id"]))
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "workspace": POLICY["workspace"],
        "environment": POLICY["environment"],
        "ledger_sha256": hashlib.sha256(raw_ledger).hexdigest(),
        "quota_sha256": hashlib.sha256(raw_quota).hexdigest(),
        "legacy_policy_sha256": hashlib.sha256(raw_policy).hexdigest(),
        "unknown_attempts": unknown,
        "active_apps": [row for row in rows if not stopped_app(row)],
        "apps": rows,
        "existing_volumes": volumes,
        "errors": [],
        "remote_compute_started": False,
        "legacy_written": False,
        "new_volume_created": False,
        "reconciliation": reconciliation,
        "initialization_requires_independent_reconciliation": True,
    }
    try:
        result["legacy_verification"] = legacy.verify(
            ledger, quota, policy, result["legacy_policy_sha256"], datetime.now(UTC)
        )
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        result["errors"].append({"kind": "legacy-validation", "type": type(error).__name__, "message": str(error)})
    write_json(output / "inventory.json", result)
    return result


def official_snapshot(workspace, first_time=None):
    observed = datetime.now(UTC)
    rows = []
    for cycle in cycle_range(first_time or observed.isoformat(), observed.isoformat()):
        value = workspace.billing.summary(cycle=cycle)
        rows.append(
            {
                "cycle": cycle,
                "start": value.start.isoformat(),
                "end": value.end.isoformat(),
                "metered_cost_usd": str(value.metered_cost),
            }
        )
    if workspace.name != POLICY["workspace"]:
        raise ValueError("Authenticated Modal workspace is not birdhackor")
    return {
        "source": SOURCE,
        "workspace": workspace.name,
        "observed_at": datetime.now(UTC).isoformat(),
        "summaries": rows,
    }


def checked_path(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not name.startswith("docs/"):
        raise ValueError("Reconciliation evidence must be an explicit repository docs path")
    following = ROOT / path
    if following.is_symlink() or not following.is_file():
        raise ValueError("Reconciliation evidence must be a regular committed file")
    return following


def load_reconciliation(name):
    path = checked_path(name)
    value = json.loads(path.read_bytes())
    for item in value["retained_bounds"]:
        for evidence in item["stop_evidence"]:
            raw = checked_path(evidence["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != evidence["sha256"]:
                raise ValueError("Independent stopping evidence changed")
    return value


def source_revision(reconciliation_path=None, output=None):
    """Pin every new runtime source, old verification dependency and evidence."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    names = [
        "scripts/omni_revision/__init__.py",
        "scripts/omni_revision/finance.py",
        "scripts/omni_revision/legacy.py",
        "scripts/omni_revision/preflight.py",
        "scripts/omni_revision/github_transport_probe.py",
        ".github/workflows/omni-revision.yml",
        "scripts/selftrained/finance.py",
        legacy.POLICY_PATH,
    ]
    if reconciliation_path is not None:
        names.append(reconciliation_path)
        reconciliation = load_reconciliation(reconciliation_path)
        names.extend(
            evidence["path"] for item in reconciliation["retained_bounds"] for evidence in item["stop_evidence"]
        )
    hashes = {}
    for name in sorted(set(names)):
        path = ROOT / name
        if path.is_symlink():
            raise ValueError("Source manifest cannot follow symlinks")
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
        raw = path.read_bytes()
        if committed != raw:
            raise ValueError("P1 sources/evidence must exactly match the selected commit")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    if output is not None:
        write_json(
            output / "source-revision.json", {"revision": revision, "files": hashes, "policy_sha256": digest(POLICY)}
        )
    return revision


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("operation", choices=["inventory", "initialize"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reconciliation", default=DEFAULT_RECONCILIATION)
    args = parser.parse_args()
    if os.environ.get("MODAL_ENVIRONMENT", "main") != POLICY["environment"]:
        raise ValueError("P1 must use the existing main environment")
    import modal

    store = GitHubLedger()
    state, _ = store.read()
    reconciliation = load_reconciliation(args.reconciliation) if args.operation == "initialize" else None
    old = inventory(modal, args.output, reconciliation)
    workspace = modal.Workspace.from_context()
    if state:
        first_cycle = state["origin"]["baseline"]["summaries"][0]["cycle"]
    else:
        old_policy = json.loads((ROOT / legacy.POLICY_PATH).read_bytes())
        first_cycle = min(old_policy["baseline"]["cycle_gross_usd"])
    first = first_cycle + "-01T00:00:00+00:00"
    snapshot = official_snapshot(workspace, first)
    rates = {key: str(value) for key, value in workspace.billing.rates().items()}
    write_json(args.output / "billing.json", snapshot)
    write_json(args.output / "rates.json", rates)
    if args.operation == "inventory":
        return
    if state is not None:
        raise ValueError("An existing round must never be reinitialized")
    revision = source_revision(args.reconciliation, args.output)
    # No paid boundary follows this CAS. Only the new GitHub budget branch changes.
    following = new_state(snapshot, old, revision)
    transaction = store.initialize(following, revision)
    write_json(args.output / "initialization.json", {"transaction": transaction, "state": following})


if __name__ == "__main__":
    main()
