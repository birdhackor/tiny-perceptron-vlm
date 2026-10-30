"""同一固定CPU backend單次量測並直接保存原始結果；不啟動Modal計算。"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from scripts.omni_revision import core_final, preflight
from scripts.omni_revision import cpu_execution as transport
from scripts.omni_revision.finance import GitHubLedger, digest, finish, reservation_identity, reserve
from scripts.omni_revision.formal_local import FROZEN, ROOT, utc
from scripts.omni_revision.github_transport_probe import REPOSITORY, api, command, find_draft
from scripts.omni_revision.research_core_inputs import check_metadata, completed_core, unpack_verified
from scripts.omni_revision.research_execution import committed
from scripts.omni_revision.research_execution import source_paths as core_and_controller_sources
from tiny_perceptron.omni_products.training_runtime import file_sha, json_file

AUDIT = ROOT / "docs/course-revision-20261029-omni/infrastructure/cpu-final-execution-audit-v2.json"
JOB = "cpu-core-final-v1"
OWN_FILES = (
    "scripts/omni_revision/cpu_final_execution.py",
    "scripts/omni_revision/legacy_download.py",
    "tests/test_omni_cpu_final_execution.py",
    ".github/workflows/omni-final-cpu.yml",
    "pyproject.toml",
    "uv.lock",
    "docs/selftrained/v2-manifest.json",
)


def source_paths():
    return sorted(set(core_and_controller_sources()) | set(OWN_FILES))


def identity(attempt):
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or attempt != f"cpu-final-{os.environ.get('GITHUB_RUN_ID')}-{os.environ.get('GITHUB_RUN_ATTEMPT')}"
        or re.fullmatch(r"cpu-final-[0-9]+-[0-9]+", attempt) is None
    ):
        raise ValueError("final只限已授權repo/main單次手動Actions工作")


def checked_sources():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    if revision != os.environ.get("GITHUB_SHA"):
        raise ValueError("final checkout與Actions SHA不同")
    hashes = {name: file_sha(ROOT / name) for name in source_paths()}
    audit = json.loads(AUDIT.read_bytes())
    if (
        audit.get("status") != "pass_for_reviewed_scope"
        or re.fullmatch(r"/root/[A-Za-z0-9_/.-]+", audit.get("reviewer_task", "")) is None
        or audit.get("files") != hashes
    ):
        raise ValueError("新增final執行／保存入口缺獨立完整來源稽核")
    for path in [*(ROOT / name for name in hashes), AUDIT]:
        committed(path, revision)
    return {"revision": revision, "files": hashes, "audit_sha256": file_sha(AUDIT)}


def prior_attempts(state):
    for row in state["attempts"]:
        try:
            purpose = json.loads(row["purpose"])
        except json.JSONDecodeError:
            continue
        if purpose.get("job") == JOB:
            raise ValueError("final工作已嘗試；保留原結果，不允許workflow重跑")


def release_identity(attempt, revision):
    return (
        "omni-final-candidate-" + attempt,
        "Owned unpublished fixed core final measurements; " + attempt + "; source " + revision,
    )


def prepare_core(revision, state):
    metadata = check_metadata(revision)
    tag, asset = completed_core(state)
    core = ROOT / "outputs/omni-formal-core-v1"
    evidence = ROOT / "outputs/omni-core-selection-v1"
    if core.exists() or evidence.exists():
        raise ValueError("全新final checkout不得覆寫既有core／selection")
    with tempfile.TemporaryDirectory(prefix="omni-final-core-") as directory:
        folder = Path(directory)
        command(
            ["gh", "release", "download", tag, "--repo", REPOSITORY, "--pattern", asset["name"], "--dir", directory]
        )
        package = folder / asset["name"]
        if package.stat().st_size != asset["bytes"] or file_sha(package) != asset["sha256"]:
            raise ValueError("final取回的core ZIP身分不同")
        unpack_verified(package, folder / "verified")
        verified = folder / "verified"
        for name in (
            "data-definition.json",
            "metrics.json",
            "formal-recipes.json",
            "codec-selection.json",
            "model-selection.json",
        ):
            if (verified / "frozen" / name).read_bytes() != (FROZEN / name).read_bytes():
                raise ValueError("core包中的原選擇與已提交manifest不同")
        shutil.copytree(verified / "core", core)
        shutil.copytree(verified / "selection-evidence", evidence)
    # 只load已核權重；不得在前置階段做任何NN預測。
    core_final.opening(core, metadata["core_selection_sha256"])
    return {**metadata, "core_asset": asset}


def legacy_inputs(output):
    """下載與離線loader核對使用兩個新程序，不共用Hub client設定。"""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("legacy-download.log", "legacy-preflight.log", "legacy-download.json", "legacy-preflight.json"):
        if (output / name).exists():
            raise FileExistsError("保留既有legacy preparation原紀錄，請使用新的工作目錄")
    process_state = {"all_training_processes_exited": True}
    transport.bounded_child(
        ["-m", "scripts.omni_revision.legacy_download", "--output", str(output)],
        output / "legacy-download.log",
        900,
        process_state,
    )
    legacy = ROOT / "outputs/omni-final-legacy-v1"
    transport.bounded_child(
        [
            "-m",
            "scripts.omni_revision.core_final",
            "legacy-preflight",
            "--output",
            str(output / "legacy-preflight.json"),
            "--data-root",
            str(legacy / "data"),
            "--old-a-root",
            str(legacy / "a"),
            "--old-b-root",
            str(legacy / "b"),
        ],
        output / "legacy-preflight.log",
        180,
        process_state,
    )


def reserve_final(attempt, output):
    identity(attempt)
    sources = checked_sources()
    output.mkdir(parents=True, exist_ok=False)
    store = GitHubLedger()
    state, blob = store.read()
    if state is None:
        raise ValueError("不能初始化或重設round")
    prior_attempts(state)
    core = prepare_core(sources["revision"], state)
    legacy_inputs(output)
    proof = transport.pricing()
    import modal

    reconciliation = preflight.load_reconciliation(preflight.DEFAULT_RECONCILIATION)
    inventory = preflight.inventory(modal, output, reconciliation)
    first = state["origin"]["baseline"]["summaries"][0]["cycle"] + "-01T00:00:00+00:00"
    snapshot = preflight.official_snapshot(modal.Workspace.from_context(), first)
    purpose = json.dumps(
        {
            "job": JOB,
            "ownership": {"attempt": attempt, "runner": "GitHub standard CPU"},
            "sources_sha256": digest(sources),
            "model_selection_sha256": core["core_selection_sha256"],
            "final_opening_sha256": core["core_final_opening_sha256"],
            "cost_proof_sha256": digest(proof),
        },
        sort_keys=True,
    )
    following = reserve(state, snapshot, inventory, attempt, transport.COMPONENTS, sources["revision"], purpose)
    json_file(output / "future-protection.json", transport.protect_core(following))
    for name, value in [
        ("sources", sources),
        ("core-inputs", core),
        ("cost-proof", proof),
        ("billing-before", snapshot),
    ]:
        json_file(output / f"{name}.json", value)
    transaction = store.write(following, blob)
    json_file(output / "reservation.json", {"transaction": transaction, "attempt": following["attempts"][-1]})
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write("reserved=true\n")


def run_final(attempt, output):
    identity(attempt)
    sources = checked_sources()
    reserved = json.loads((output / "reservation.json").read_bytes())["attempt"]
    state, _ = GitHubLedger().read()
    row = next(row for row in state["attempts"] if row["attempt_id"] == attempt)
    if (
        row["status"] != "reserved"
        or reservation_identity(row) != reservation_identity(reserved)
        or row["revision"] != sources["revision"]
    ):
        raise ValueError("final預約身分已改")
    # 放在try/export之前；第二次進入不能覆寫第一次的原始receipt或重跑任何part。
    with (output / "final-run-started.json").open("x", encoding="utf-8") as stream:
        json.dump({"attempt": attempt, "started_at": utc(), "source_revision": sources["revision"]}, stream)
    tag, body = release_identity(attempt, sources["revision"])
    receipt = {
        "schema": "omni-cpu-final-execution-v1",
        "attempt": attempt,
        "started_at": utc(),
        "source_revision": sources["revision"],
        "success": False,
        "no_modal_compute_created": True,
        "no_training_performed": True,
        "all_training_processes_exited": True,
        "release_tag": tag,
        "assets": [],
        "completed_parts": [],
    }
    try:
        if find_draft(tag, body) is not None:
            raise ValueError("final唯一nonce已存在，禁止重跑")
        json_file(output / "release-ownership.json", {"body": body, "tag": tag, "create_started": True})
        created = api(
            "releases",
            "POST",
            {
                "tag_name": tag,
                "target_commitish": sources["revision"],
                "name": "Unpublished fixed Omni final measurements",
                "body": body,
                "draft": True,
                "prerelease": True,
            },
        )
        receipt["release_id"] = created["id"]
        json_file(output / "release-ownership.json", {"body": body, "tag": tag, "release_id": created["id"]})
        selection = file_sha(FROZEN / "model-selection.json")
        for part in ("av", "sound", "choice", "integration"):
            arguments = [
                "-m",
                "scripts.omni_revision.core_final",
                part,
                "--selection-sha256",
                selection,
                "--output",
                str(output / "measurements" / part),
            ]
            if part == "integration":
                legacy = ROOT / "outputs/omni-final-legacy-v1"
                arguments.extend(
                    [
                        "--data-root",
                        str(legacy / "data"),
                        "--old-a-root",
                        str(legacy / "a"),
                        "--old-b-root",
                        str(legacy / "b"),
                    ]
                )
            transport.bounded_child(arguments, output / f"{part}.log", 1500, receipt)
            receipt["completed_parts"].append(part)
            json_file(output / "runtime-progress.json", receipt)
        transport.bounded_child(
            [
                "-m",
                "scripts.omni_revision.core_report",
                "--results",
                str(output / "measurements"),
                "--output",
                str(output / "final-report.json"),
            ],
            output / "report.log",
            120,
            receipt,
        )
        receipt["success"] = True
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error)[:1000])
    finally:
        try:
            paths = [
                ("measurements", output / "measurements"),
                ("frozen", FROZEN),
                ("execution-sources.json", output / "sources.json"),
                ("core-inputs.json", output / "core-inputs.json"),
                ("final-report.json", output / "final-report.json"),
            ]
            package = transport.package(output, paths, "final-results.zip")
            receipt["package"] = {"bytes": package.stat().st_size, "sha256": file_sha(package)}
            owned = api(f"releases/{receipt['release_id']}") if receipt.get("release_id") else find_draft(tag, body)
            if (
                owned is None
                or owned.get("draft") is not True
                or owned.get("tag_name") != tag
                or owned.get("body") != body
                or owned.get("target_commitish") != sources["revision"]
                or (receipt.get("release_id") is not None and owned["id"] != receipt["release_id"])
            ):
                raise ValueError("final結果的精確owned draft無法核實，匯出失敗")
            receipt["release_id"] = owned["id"]
            json_file(
                output / "release-ownership.json",
                {"body": body, "tag": tag, "release_id": owned["id"], "create_started": True},
            )
            receipt["assets"].append(transport.upload_owned(tag, body, owned["id"], package))
        except BaseException as error:
            receipt.update(success=False, export_error_type=type(error).__name__)
        receipt["finished_at"] = utc()
        json_file(output / "cpu-final-receipt.json", receipt)
    if not receipt["success"]:
        raise RuntimeError("final工作不完整；原始partial已保留，不允許盲目重跑")


def finish_final(attempt, output):
    identity(attempt)
    sources = checked_sources()
    reserved = json.loads((output / "reservation.json").read_bytes())["attempt"]
    receipt = json.loads((output / "cpu-final-receipt.json").read_bytes())
    bounds = transport.artifact_bound(output)
    ownership = json.loads((output / "release-ownership.json").read_bytes())
    store = GitHubLedger()
    state, blob = store.read()
    rows = [row for row in state["attempts"] if row["attempt_id"] == attempt]
    purpose = json.loads(reserved["purpose"])
    tag, body = release_identity(attempt, sources["revision"])
    if (
        len(rows) != 1
        or rows[0]["status"] != "reserved"
        or reservation_identity(rows[0]) != reservation_identity(reserved)
        or reserved["attempt_id"] != attempt
        or reserved["revision"] != sources["revision"]
        or sources["revision"] != os.environ.get("GITHUB_SHA")
        or purpose.get("job") != JOB
        or purpose.get("ownership", {}).get("attempt") != attempt
        or purpose.get("sources_sha256") != digest(sources)
        or receipt.get("schema") != "omni-cpu-final-execution-v1"
        or receipt.get("attempt") != attempt
        or receipt.get("source_revision") != sources["revision"]
        or receipt.get("release_tag") != tag
        or ownership.get("tag") != tag
        or ownership.get("body") != body
        or type(ownership.get("release_id")) is not int
        or ownership["release_id"] <= 0
        or receipt.get("release_id") != ownership["release_id"]
    ):
        raise ValueError("final收據／預約／Actions來源／唯一Release身分不同")
    owned = (
        api(f"releases/{receipt['release_id']}")
        if receipt.get("release_id")
        else find_draft(receipt["release_tag"], ownership["body"])
    )
    release_ok = (
        owned is not None
        and owned.get("draft") is True
        and owned.get("id") == ownership["release_id"]
        and owned.get("body") == body
        and owned.get("tag_name") == tag
        and owned.get("target_commitish") == sources["revision"]
        and {(row["id"], row["name"], row["size"]) for row in owned["assets"]}
        == {(row["id"], row["name"], row["bytes"]) for row in receipt["assets"]}
        and len(receipt["assets"]) == 1
        and all(row.get("roundtrip_verified") is True for row in receipt["assets"])
    )
    record = api(f"actions/runs/{os.environ['GITHUB_RUN_ID']}/artifacts?per_page=100")
    artifacts = record["artifacts"]
    expected = "omni-final-" + os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    artifact_ok = (
        record["total_count"] == len(artifacts) == 1
        and artifacts[0]["name"] == expected
        and 0 < artifacts[0]["size_in_bytes"] <= transport.MAX_ARTIFACT
        and artifacts[0].get("expired") is False
        and 0
        < (
            datetime.fromisoformat(artifacts[0]["expires_at"].replace("Z", "+00:00"))
            - datetime.fromisoformat(artifacts[0]["created_at"].replace("Z", "+00:00"))
        ).total_seconds()
        <= 7 * 86400 + 60
        and os.environ.get("OMNI_ARTIFACT_OUTCOME") == "success"
    )
    receipt.update(
        success=receipt["success"] and release_ok and artifact_ok,
        compute_stopped=receipt["all_training_processes_exited"],
        resources_verified=release_ok
        and artifact_ok
        and receipt["all_training_processes_exited"]
        and receipt["no_modal_compute_created"],
        cleanup_checks={
            "all_final_children_exited": receipt["all_training_processes_exited"],
            "no_modal_compute_created": receipt["no_modal_compute_created"],
            "owned_final_results_retained": release_ok,
            "artifact_provider_limits_verified": artifact_ok,
        },
        artifact=bounds,
        artifact_api_records=[
            {key: row[key] for key in ("id", "name", "size_in_bytes", "created_at", "expires_at")} for row in artifacts
        ],
    )
    following = finish(state, attempt, receipt)
    transaction = store.write(following, blob)
    json_file(output / "finish-transaction.json", {"transaction": transaction, "attempt": following["attempts"][-1]})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("operation", choices=("reserve", "run", "bound", "finish", "legacy"))
    parser.add_argument("--attempt-id")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "legacy":
        legacy_inputs(args.output)
    elif args.operation == "bound":
        print(json.dumps(transport.artifact_bound(args.output)))
    else:
        {"reserve": reserve_final, "run": run_final, "finish": finish_final}[args.operation](
            args.attempt_id, args.output
        )


if __name__ == "__main__":
    main()
