"""Actions標準CPU的正式工作與耐久成品；不建立任何Modal計算資源。"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from scripts.omni_revision import preflight
from scripts.omni_revision.experiment_execution import STORAGE_URL, PricingText, protect_core, storage_quote
from scripts.omni_revision.finance import GitHubLedger, digest, finish, reservation_identity, reserve
from scripts.omni_revision.formal_local import AUDIT, FROZEN, ROOT, checked_job
from scripts.omni_revision.github_transport_probe import REPOSITORY, api, command, find_draft
from tiny_perceptron.omni_products.training_runtime import file_sha, json_file

COMPONENTS = {"compute": "0", "startup_idle": "0", "build": "0", "egress": "0", "storage_cleanup": "0.10"}
MAX_PACKAGE = 128 * 1024**2
MAX_ARTIFACT = 256 * 1024**2
MAX_CONTROL = 16 * 1024**2
MAX_LOG = 256 * 1024


def identity(attempt):
    if (
        re.fullmatch(r"cpu-core-[0-9]+-[0-9]+", attempt) is None
        or os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or attempt != f"cpu-core-{os.environ.get('GITHUB_RUN_ID')}-{os.environ.get('GITHUB_RUN_ATTEMPT')}"
    ):
        raise ValueError("只限已批准repo的main手動工作與唯一run/attempt")


def checked_sources():
    document = json.loads((FROZEN / "formal-recipes.json").read_bytes())
    checked_job(FROZEN, "codec")
    names = set(document["execution_sources_sha256"])
    names.update(json.loads((FROZEN / "metrics.json").read_bytes())["measurement_sources_sha256"])
    names.update(f"tiny_perceptron/omni_products/{name}" for name in document["implementation_sha256"])
    names.update(
        str((FROZEN / name).relative_to(ROOT))
        for name in ("data-definition.json", "metrics.json", "formal-recipes.json")
    )
    names.add(str(AUDIT.relative_to(ROOT)))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("固定revision不合法")
    if revision != os.environ.get("GITHUB_SHA"):
        raise ValueError("checkout與本次Actions的固定SHA不同")
    for name in names:
        if (ROOT / name).is_symlink() or subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT) != (
            ROOT / name
        ).read_bytes():
            raise ValueError("實際執行來源必須已提交且逐bytes等同固定revision")
    return {"revision": revision, "files": {name: file_sha(ROOT / name) for name in sorted(names)}}


def pricing():
    repository = json.loads(command(["gh", "api", f"repos/{REPOSITORY}"]))
    if repository.get("private") is not False or repository.get("full_name") != REPOSITORY:
        raise ValueError("標準runner免費配方只適用此公開repo")
    with urllib.request.urlopen(STORAGE_URL, timeout=30) as response:
        if response.url != STORAGE_URL:
            raise ValueError("費率來源轉址")
        raw = response.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2:
        raise ValueError("費率來源超界")
    parser = PricingText()
    parser.feed(raw.decode())
    text = " ".join(parser.parts)
    excerpt = "The use of standard GitHub-hosted runners is free: In public repositories"
    if excerpt not in text:
        raise ValueError("無法確認標準公開runner的免費條件")
    return {
        "storage": storage_quote(raw),
        "public_standard_runner_compute": {
            "source_url": STORAGE_URL,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "excerpt": excerpt,
            "repository_private": False,
            "runner": "ubuntu-latest standard, not larger runner",
            "maximum_minutes": 300,
            "usd_compute_bound": "0",
        },
        "artifact_max_bytes": MAX_ARTIFACT,
        "retention_days": 7,
        "storage_and_cleanup_bound": "0.10",
    }


def reserve_cpu(attempt, output):
    identity(attempt)
    sources = checked_sources()
    if any((FROZEN / name).exists() for name in ("codec-selection.json", "model-selection.json", "final-opening.json")):
        raise ValueError("正式產物選定或final已開，不能重新開始同round訓練")
    output.mkdir(parents=True, exist_ok=False)
    store = GitHubLedger()
    state, blob = store.read()
    if state is None:
        raise ValueError("不得重設既有round")
    for row in state["attempts"]:
        try:
            purpose = json.loads(row["purpose"])
        except json.JSONDecodeError:
            continue
        if purpose.get("job") == "cpu-core-formal":
            raise ValueError("十個核心正式工作已嘗試，不允許workflow盲目重跑")
    proof = pricing()
    import modal

    reconciliation = preflight.load_reconciliation(preflight.DEFAULT_RECONCILIATION)
    inventory = preflight.inventory(modal, output, reconciliation)
    first = state["origin"]["baseline"]["summaries"][0]["cycle"] + "-01T00:00:00+00:00"
    snapshot = preflight.official_snapshot(modal.Workspace.from_context(), first)
    purpose = json.dumps(
        {
            "job": "cpu-core-formal",
            "ownership": {"attempt": attempt, "runner": "GitHub standard CPU"},
            "sources_sha256": digest(sources),
            "formal_recipe_sha256": file_sha(FROZEN / "formal-recipes.json"),
            "cost_proof_sha256": digest(proof),
        },
        sort_keys=True,
    )
    following = reserve(state, snapshot, inventory, attempt, COMPONENTS, sources["revision"], purpose)
    json_file(output / "core-protection.json", protect_core(following))
    for name, value in (("sources", sources), ("cost-proof", proof), ("billing-before", snapshot)):
        json_file(output / f"{name}.json", value)
    transaction = store.write(following, blob)
    json_file(output / "reservation.json", {"transaction": transaction, "attempt": following["attempts"][-1]})


def bounded_child(arguments, log, seconds, state):
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"GH_TOKEN", "GITHUB_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}
    }
    with log.open("wb") as stream:
        state["all_training_processes_exited"] = False
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, *arguments], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
            deadline = time.monotonic() + seconds
            while process.poll() is None:
                if log.stat().st_size > MAX_LOG:
                    raise RuntimeError("核心工作log超出256KiB上界")
                if time.monotonic() > deadline:
                    raise TimeoutError("核心工作外層絕對deadline")
                time.sleep(0.2)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=30)
                state["all_training_processes_exited"] = process.poll() is not None
    raw = log.read_bytes()
    if len(raw) > MAX_LOG:
        raise RuntimeError("核心工作log超界")
    print(raw.decode("utf-8", errors="replace"), end="", flush=True)
    if process.returncode:
        raise RuntimeError("核心子工作失敗；保存原log，不自動重跑")


def package(output, source_paths, name):
    files = []
    for prefix, path in source_paths:
        if path.is_dir():
            files.extend((f"{prefix}/{p.relative_to(path)}", p) for p in sorted(path.rglob("*")) if p.is_file())
        elif path.is_file():
            files.append((prefix, path))
    if any(path.is_symlink() for _, path in files) or sum(path.stat().st_size for _, path in files) > MAX_ARTIFACT:
        raise ValueError("成品來源超過256MiB或含symlink")
    manifest = {key: {"bytes": path.stat().st_size, "sha256": file_sha(path)} for key, path in files}
    destination = output / name
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for key, path in files:
            archive.write(path, key)
        archive.writestr("package-file-manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
    if destination.stat().st_size > MAX_PACKAGE:
        raise ValueError("成品zip超過128MiB")
    return destination


def upload_owned(tag, body, release_id, path):
    before = api(f"releases/{release_id}")
    if before.get("draft") is not True or before.get("tag_name") != tag or before.get("body") != body:
        raise ValueError("草稿ownership不符，禁止寫入")
    if any(item["name"] == path.name for item in before["assets"]):
        raise ValueError("不可覆寫既有成品")
    command(["gh", "release", "upload", tag, str(path), "--repo", REPOSITORY])
    after = api(f"releases/{release_id}")
    matches = [item for item in after["assets"] if item["name"] == path.name]
    if len(matches) != 1 or matches[0]["size"] != path.stat().st_size:
        raise ValueError("成品上傳身份／大小不符")
    asset = matches[0]
    if asset.get("digest") not in (None, "sha256:" + file_sha(path)):
        raise ValueError("GitHub遠端成品SHA與本機不同")
    with tempfile.TemporaryDirectory(prefix="omni-core-readback-") as temporary:
        command(["gh", "release", "download", tag, "--repo", REPOSITORY, "--pattern", path.name, "--dir", temporary])
        received = Path(temporary) / path.name
        if received.stat().st_size != path.stat().st_size or file_sha(received) != file_sha(path):
            raise ValueError("GitHub成品實際下載的bytes／SHA不同")
    return {
        "id": asset["id"],
        "name": asset["name"],
        "bytes": asset["size"],
        "sha256": file_sha(path),
        "roundtrip_verified": True,
    }


def run_cpu(attempt, output):
    identity(attempt)
    sources = checked_sources()
    reserved = json.loads((output / "reservation.json").read_bytes())["attempt"]
    if reserved["attempt_id"] != attempt or reserved["revision"] != sources["revision"]:
        raise ValueError("CPU預約來源不同")
    core = ROOT / "outputs/omni-formal-core-v1"
    if core.exists():
        raise ValueError("不覆寫或重跑既有formal output")
    document = json.loads((FROZEN / "formal-recipes.json").read_bytes())
    tag = "omni-core-candidate-" + attempt
    body = "Owned unpublished core weights and full optimizer/RNG; " + attempt + "; source " + sources["revision"]
    result = {
        "schema": "omni-cpu-core-execution-v1",
        "attempt": attempt,
        "success": False,
        "final_opened": False,
        "no_modal_compute_created": True,
        "all_training_processes_exited": True,
        "release_tag": tag,
        "assets": [],
        "source_revision": sources["revision"],
        "started_at": datetime.now(UTC).isoformat(),
    }
    try:
        if find_draft(tag, body) is not None:
            raise ValueError("唯一草稿nonce已存在，不重跑")
        json_file(output / "release-ownership.json", {**result, "body": body, "create_started": True})
        created = api(
            "releases",
            "POST",
            {
                "tag_name": tag,
                "target_commitish": sources["revision"],
                "name": "Unpublished Omni core training candidate",
                "body": body,
                "draft": True,
                "prerelease": True,
            },
        )
        result["release_id"] = created["id"]
        json_file(output / "release-ownership.json", {**result, "body": body})
        order = ["codec", "talker", *[name for name in document["job_order"] if name not in {"codec", "talker"}]]
        for job in order:
            print(json.dumps({"starting_formal_job": job, "final_opened": False}), flush=True)
            cap = document["jobs"][job]["training"]["training_seconds"] + 330
            bounded_child(["-m", "scripts.omni_revision.formal_local", job], output / f"{job}.log", cap, result)
            if job == "codec":
                bounded_child(
                    ["-m", "scripts.omni_revision.core_selection", "freeze-codec"],
                    output / "codec-selection.log",
                    60,
                    result,
                )
            asset = package(output, [(f"core/{job}", core / job)], job + ".zip")
            result["assets"].append(upload_owned(tag, body, created["id"], asset))
            asset.unlink()
            json_file(output / "runtime-progress.json", result)
        bounded_child(
            ["-m", "scripts.omni_revision.core_selection", "select-models"], output / "model-selection.log", 300, result
        )
        result["success"] = True
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error)[:1000])
    finally:
        paths = [("core", core), ("frozen", FROZEN), ("selection-evidence", ROOT / "outputs/omni-core-selection-v1")]
        try:
            archive = package(output, paths, "core-checkpoints.zip")
            result["package"] = {"bytes": archive.stat().st_size, "sha256": file_sha(archive)}
            if result.get("release_id"):
                result["assets"].append(upload_owned(tag, body, result["release_id"], archive))
        except BaseException as error:
            result.update(success=False, export_error_type=type(error).__name__)
        result["finished_at"] = datetime.now(UTC).isoformat()
        json_file(output / "cpu-receipt.json", result)
    if not result["success"]:
        raise RuntimeError("正式CPU核心工作不完整；保留partial及原receipt，不自動重跑")


def artifact_bound(output):
    files = [path for path in output.rglob("*") if path.is_file()]
    controls = sum(path.stat().st_size for path in files if path.suffix != ".zip")
    total = sum(path.stat().st_size for path in files)
    packages = sum(path.stat().st_size for path in files if path.suffix == ".zip")
    if (
        any(path.is_symlink() for path in files)
        or controls > MAX_CONTROL
        or total > MAX_ARTIFACT
        or packages > MAX_PACKAGE
    ):
        raise ValueError("CPU artifact超過control16MiB／aggregate256MiB上界")
    return {"total_bytes": total, "control_bytes": controls, "retention_days": 7}


def finish_cpu(attempt, output):
    identity(attempt)
    reserved = json.loads((output / "reservation.json").read_bytes())["attempt"]
    receipt = json.loads((output / "cpu-receipt.json").read_bytes())
    bounds = artifact_bound(output)
    ownership = json.loads((output / "release-ownership.json").read_bytes())
    owned = (
        api(f"releases/{receipt['release_id']}")
        if receipt.get("release_id")
        else find_draft(receipt["release_tag"], ownership["body"])
    )
    release_verified = (
        owned is not None
        and owned.get("draft") is True
        and owned.get("tag_name") == receipt["release_tag"]
        and owned.get("body") == ownership["body"]
        and {(asset["id"], asset["name"], asset["size"]) for asset in owned["assets"]}
        == {(asset["id"], asset["name"], asset["bytes"]) for asset in receipt["assets"]}
        and all(asset.get("roundtrip_verified") is True for asset in receipt["assets"])
    )
    publication = os.environ.get("OMNI_ARTIFACT_OUTCOME")
    artifact_verified = False
    try:
        records = api(f"actions/runs/{os.environ['GITHUB_RUN_ID']}/artifacts?per_page=100")
        items = records["artifacts"]
        if records["total_count"] != len(items):
            raise ValueError("此run artifact超過單頁上界")
        expected_name = "omni-core-" + os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
        artifact_verified = (
            all(
                item["name"] == expected_name
                and item["size_in_bytes"] <= MAX_ARTIFACT
                and (
                    datetime.fromisoformat(item["expires_at"].replace("Z", "+00:00"))
                    - datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
                ).total_seconds()
                <= 7 * 86400 + 60
                for item in items
            )
            and len(items) <= 1
            and (publication == "skipped" or (publication == "success" and len(items) == 1))
        )
        receipt["artifact_api_records"] = [
            {key: item[key] for key in ("id", "name", "size_in_bytes", "created_at", "expires_at")} for item in items
        ]
    except Exception as error:
        receipt["artifact_verification_error_type"] = type(error).__name__
    receipt["success"] = receipt["success"] and publication == "success" and release_verified and artifact_verified
    receipt.update(
        compute_stopped=receipt["all_training_processes_exited"],
        resources_verified=release_verified
        and artifact_verified
        and receipt["all_training_processes_exited"]
        and receipt["no_modal_compute_created"],
        cleanup_checks={
            "no_modal_compute_created": receipt["no_modal_compute_created"],
            "all_training_processes_exited": receipt["all_training_processes_exited"],
            "artifact_retention_and_size_bounded": artifact_verified,
            "weights_intentionally_retained_in_verified_owned_draft": release_verified,
        },
        artifact=bounds,
    )
    store = GitHubLedger()
    state, blob = store.read()
    matches = [row for row in state["attempts"] if row["attempt_id"] == attempt]
    if len(matches) != 1 or reservation_identity(matches[0]) != reservation_identity(reserved):
        raise ValueError("CPU預約身分已改")
    final = finish(state, attempt, receipt)
    transaction = store.write(final, blob)
    json_file(output / "finish-transaction.json", {"transaction": transaction, "attempt": final["attempts"][-1]})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("operation", choices=("reserve", "run", "bound", "finish"))
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "reserve":
        reserve_cpu(args.attempt_id, args.output)
    elif args.operation == "run":
        run_cpu(args.attempt_id, args.output)
    elif args.operation == "finish":
        finish_cpu(args.attempt_id, args.output)
    else:
        print(json.dumps(artifact_bound(args.output)))


if __name__ == "__main__":
    main()
