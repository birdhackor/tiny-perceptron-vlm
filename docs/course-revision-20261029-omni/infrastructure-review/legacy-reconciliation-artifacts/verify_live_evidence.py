"""Independent read-only verification of the exact live legacy evidence set.

Only this review's JSON result is written. No GH, Modal, ledger, quota, or
reconciliation mutation occurs here. Full logs/source stay outside ordinary Git.
"""

import ast
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.omni_revision import finance, legacy

HERE = Path(__file__).resolve().parent
COURSE = ROOT / "docs/course-revision-20261029-omni"
LIVE = Path("/workspace/work/omni-p1-live-inventory")
META = Path("/workspace/work/omni-legacy-github")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def verify_zip(path, metadata, exact_files):
    assert sha(path.read_bytes()) == metadata["digest"].removeprefix("sha256:")
    with zipfile.ZipFile(path) as archive:
        for name, raw in exact_files.items():
            candidates = [item for item in archive.namelist() if item == name or item.endswith("/" + name)]
            assert len(candidates) == 1, (path, name, candidates)
            assert archive.read(candidates[0]) == raw, (path, name)
    return {"path": str(path), "artifact_id": metadata["id"], "sha256": sha(path.read_bytes()),
            "verified_members": sorted(exact_files)}


def main():
    index = read(COURSE / "legacy-stop-evidence.json")
    assert sha((COURSE / "legacy-stop-evidence.json").read_bytes()) == "2eb097d2d96418fc52cbce383059c71d35829636cd5653298f84a5a6e685689f"
    inventory = read(LIVE / "inventory.json")
    ledger, quota, policy = [read(LIVE / name) for name in ("legacy-budget.json", "legacy-gross-quota.json", "legacy-policy.json")]
    for field, name in [("ledger_sha256", "legacy-budget.json"), ("quota_sha256", "legacy-gross-quota.json"),
                        ("legacy_policy_sha256", "legacy-policy.json")]:
        assert sha((LIVE / name).read_bytes()) == inventory[field] == index["source_inventory"][field]
    assert (LIVE / "legacy-policy.json").read_bytes() == (ROOT / legacy.POLICY_PATH).read_bytes()
    assert not inventory["active_apps"] and not inventory["unknown_attempts"] and not inventory["errors"]
    verified = legacy.verify(ledger, quota, policy, inventory["legacy_policy_sha256"], datetime.now(UTC))
    assert verified == inventory["legacy_verification"]
    selected = quota["attempts"] + policy["carry_in_bounds"]
    ids = [item["run_id"] for item in selected]
    assert len(ids) == len(set(ids)) == 14
    exact_carry = sum((Decimal(str(item["reserved_usd"])) for item in selected), Decimal(0))
    assert exact_carry == Decimal("4.73")
    records = {row["run_id"]: row for row in ledger["reservations"]}
    assert len(records) == len(ledger["reservations"])
    assert Decimal(ledger["reserved_total_usd"]) == Decimal("52.59")
    assert sum((Decimal(item["reserved_usd"]) for item in records.values()), Decimal(0)) == Decimal("52.59")
    assert {item["run_id"] for item in index["attempts"]} == set(ids)
    for item in index["evidence_files"]:
        raw = (ROOT / item["path"]).read_bytes()
        assert sha(raw) == item["sha256"] and len(raw) == item["bytes"]

    log_index = {item["run_id"]: item for item in read(COURSE / "legacy-evidence/controller-log-excerpts.json")["logs"]}
    source_index = read(COURSE / "legacy-evidence/controller-source-excerpts.json")
    excerpts = {item["sha256"]: item["text"] for item in source_index["excerpts"]}
    source_results = []
    for blob in source_index["blobs"]:
        if blob["retrieval"]["kind"] == "local-git-object":
            raw = subprocess.check_output(["git", "show", f"{blob['revision']}:{blob['path']}"], cwd=ROOT)
        else:
            raw = Path(blob["retrieval"]["cached_full_source_path"]).read_bytes()
            assert read(HERE / "publish-source-recheck.json")["sha256"] == sha(raw)
        assert sha(raw) == blob["raw_sha256"] == blob["ledger_source_sha256"]
        lines = raw.decode().splitlines(keepends=True)
        for excerpt in blob["excerpts"]:
            text = "".join(lines[excerpt["start_line"]-1:excerpt["end_line_inclusive"]])
            assert sha(text.encode()) == excerpt["excerpt_sha256"]
            assert text == excerpts[excerpt["excerpt_sha256"]]
        run_calls = [node for node in ast.walk(ast.parse(raw)) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                     and node.func.value.id == "app" and node.func.attr == "run"]
        assert all(not node.args and not node.keywords for node in run_calls)
        source_results.append({"revision": blob["revision"], "path": blob["path"], "sha256": sha(raw),
                               "app_run_lines": [node.lineno for node in run_calls]})

    official = read(HERE / "official-rechecks.json")
    job_sets = {item["workflow_run_id"]: item["jobs"] for item in official["jobs"]}
    metadata_index = {item["run_id"]: item for item in read(COURSE / "legacy-evidence/actions-metadata-projections.json")["runs"]}
    attempts = []
    for item in index["attempts"]:
        run_id = item["run_id"]
        record = records[run_id]
        reference = item["original_receipt"]
        raw = (ROOT / reference["path"]).read_bytes()
        assert sha(raw) == reference["sha256"]
        receipt = json.loads(raw)
        for part in reference["json_pointer"].split("/")[1:]:
            receipt = receipt[part.replace("~1", "/").replace("~0", "~")]
        assert sha(canonical(receipt)) == record["receipt_sha256"] == reference["canonical_sha256"]
        assert receipt["run_id"] == run_id and receipt["status"] == "completed"
        assert receipt.get("revision", receipt.get("code_revision")) == record["revision"] == item["source_revision"]
        source_path = item["source_binding"].split(":", 1)[1]
        source = next(row for row in source_results if row["revision"] == record["revision"] and row["path"] == source_path)
        assert source["sha256"] == record["source_sha256"][source_path]
        meta = metadata_index[run_id]
        run_num = item["controller"]["workflow_run_id"]
        assert meta["run"]["id"] == run_num and meta["run"]["head_sha"] == record["revision"]
        assert sha((META / f"{run_num}-run-raw.json").read_bytes()) == meta["run_raw_sha256"]
        assert sha((META / f"{run_num}-jobs-raw.json").read_bytes()) == meta["jobs_raw_sha256"]
        job = next(row for row in job_sets[run_num] if row["id"] == item["controller"]["job_id"])
        assert job["status"] == "completed" and job["conclusion"] == item["controller"]["job_conclusion"]
        log = log_index[run_id]
        raw_log = Path(log["full_log_workspace_path"]).read_bytes()
        assert sha(raw_log) == log["full_decoded_utf8_log_sha256"] and len(raw_log) == log["full_log_bytes"]
        lines = raw_log.decode().splitlines()
        assert all(lines[excerpt["line"]-1] == excerpt["text"] for excerpt in log["excerpts"])
        job_metadata = next(row for row in meta["jobs"] if row["id"] == job["id"])
        completed_at = datetime.fromisoformat(job_metadata["completed_at"])
        final_log_time = datetime.fromisoformat(log["excerpts"][-1]["text"].split(" ", 1)[0])
        assert datetime.fromisoformat(record["finished_at"]) <= final_log_time <= completed_at
        for extra in item["supplemental"]:
            raw_extra = (ROOT / extra["path"]).read_bytes()
            assert sha(raw_extra) == extra["sha256"]
            controller = json.loads(raw_extra)
            assert controller["status"] == "failed" and controller["error_type"] == "DeserializationError"
            assert controller["function_call_id"] == record["function_call_id"]
        attempts.append({"run_id": run_id, "bound_usd": record["reserved_usd"],
                         "remote_receipt_status": receipt["status"], "receipt_canonical_sha256": record["receipt_sha256"],
                         "controller_conclusion": job["conclusion"], "job_id": job["id"],
                         "explicit_app_completed_in_log": any("App completed." in line for line in lines),
                         "source_sha256": source["sha256"], "full_log_sha256": sha(raw_log),
                         "ledger_finished_at": record["finished_at"],
                         "final_controller_log_at": final_log_time.isoformat(),
                         "controller_completed_at": completed_at.isoformat(),
                         "remote_finished_before_controller_terminal": True})

    zip_results = []
    for entry in official["artifacts"]:
        directory = LIVE if entry["channel"] == "inventory-artifacts" else Path("/workspace/work/omni-p1-live-transport")
        exact = {path.name: path.read_bytes() for path in directory.glob("*.json")}
        zip_results.append(verify_zip(directory / "artifact.zip", entry["artifacts"][0], exact))
    recovered = read(HERE / "recovered-artifact-rechecks.json")
    for entry in recovered["artifacts"]:
        run_num = entry["run_id"]
        if run_num == 37603079766:
            directory = Path("/workspace/work/omni-legacy-card-37603079766")
            exact = {"receipt.json": (ROOT / index["attempts"][7]["original_receipt"]["path"]).read_bytes()}
        else:
            directory = Path(f"/workspace/work/omni-legacy-controller-{run_num}")
            extra = next(item for item in index["attempts"] if item["controller"]["workflow_run_id"] == run_num)["supplemental"][0]
            exact = {"runner-result.json": (ROOT / extra["path"]).read_bytes()}
        zip_results.append(verify_zip(directory / "artifact.zip", entry["artifact"], exact))
    result = {"reviewer_task": "/root/omni_p1_infra_review", "checked_at": datetime.now(UTC).isoformat(),
              "legacy_raw_sha256": {key: inventory[key] for key in ("ledger_sha256", "quota_sha256", "legacy_policy_sha256")},
              "inventory_sha256": sha((LIVE / "inventory.json").read_bytes()),
              "legacy_verification_sha256": finance.digest(verified),
              "existing_volumes_sha256": finance.digest(inventory["existing_volumes"]),
              "retained_count": len(ids), "exact_carry_usd": str(exact_carry),
              "audit_total_usd": ledger["reserved_total_usd"], "historical_audit_excluded_from_carry": True,
              "attempts": attempts, "sources": source_results, "zip_artifacts": zip_results,
              "no_remote_mutation": True, "all_bounds_retained": True,
              "limits": ["This script verifies identities and arithmetic; the independent report supplies lifecycle judgment.",
                         "Snapshot observation is historical, not a fresh initialization baseline.",
                         "No per-run settled costs, invoice finality, measured Volume sizes or perpetual storage bound."]}
    (HERE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"retained_count": len(ids), "carry_usd": str(exact_carry), "receipt_matches": len(attempts),
                      "source_blobs": len(source_results), "zip_digests_and_members": len(zip_results),
                      "app_completed_logs": sum(row["explicit_app_completed_in_log"] for row in attempts)}, indent=2))


if __name__ == "__main__":
    main()
