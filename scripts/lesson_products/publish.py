"""只發布已評測且SHA固定的B/C包；沿用同一gross額度，不啟動GPU。"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from scripts.lesson_products.modal_runner import POLICY_PATH, POLICY_SHA, check_source_files, confirm_stopped
from scripts.lesson_products.publication import redact_error, sha, unpack_verified, verify_release
from scripts.selftrained.finance import official_snapshot
from scripts.selftrained.modal_runner import (
    canonical_sha,
    finish_entry,
    live_snapshot,
    read_live_ledger,
    reserve_entry,
    safe_name,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
LEDGER, QUOTA = Path("/course/budget.json"), Path("/course/gross-quota.json")


def inputs(revision, spec_path):
    if (
        not re.fullmatch(r"[a-f0-9]{40}", revision)
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip() != revision
    ):
        raise ValueError("發布需實際checkout的精確commit")
    path = Path(spec_path)
    if path.parent.as_posix() != "docs/lesson-products/jobs" or path.suffix != ".json" or ".." in path.parts:
        raise ValueError("發布spec需在已提交job目錄內")
    spec = json.loads((ROOT / path).read_bytes())
    expected = {
        "version",
        "stage",
        "job_kind",
        "wall_seconds",
        "gross_quota_policy",
        "asset_id",
        "release_id",
        "release_commit",
        "asset_name",
        "asset_bytes",
        "transport_sha256",
        "data_manifest_sha256",
        "prefix",
        "files",
    }
    if (
        set(spec) != expected
        or spec["version"] != 1
        or spec["stage"] != "release"
        or spec["job_kind"] != "lesson_products_release"
        or spec["wall_seconds"] != 600
    ):
        raise ValueError("不是固定600秒CPU的成品發布描述")
    if (
        type(spec["asset_id"]) is not int
        or spec["asset_id"] <= 0
        or not re.fullmatch(r"[a-f0-9]{64}", spec["transport_sha256"])
        or not re.fullmatch(r"lesson-products/[A-Za-z0-9_-]{1,80}", spec["prefix"])
    ):
        raise ValueError("發布來源asset／transport／新prefix不合法")
    if (
        type(spec["release_id"]) is not int
        or spec["release_id"] <= 0
        or not re.fullmatch(r"[a-f0-9]{40}", spec["release_commit"])
        or not re.fullmatch(r"[A-Za-z0-9_-]+\.tar\.gz", spec["asset_name"])
        or type(spec["asset_bytes"]) is not int
        or not 0 < spec["asset_bytes"] <= 64 * 1024 * 1024
    ):
        raise ValueError("發布transport需固定draft parent、commit、檔名與大小")
    if spec["gross_quota_policy"] != {"path": POLICY_PATH, "sha256": POLICY_SHA}:
        raise ValueError("發布不可提高額度或更換gross政策")
    names = [
        "scripts/lesson_products/publish.py",
        "scripts/lesson_products/publication.py",
        "scripts/lesson_products/modal_runner.py",
        "scripts/lesson_products/checkpoints.py",
        "scripts/selftrained/modal_runner.py",
        "scripts/selftrained/finance.py",
        POLICY_PATH,
        "docs/lesson-products/data-v2-manifest.json",
        ".github/workflows/lesson-products-publish.yml",
        spec_path,
    ]
    hashes = {}
    for name in names:
        raw = (ROOT / name).read_bytes()
        if raw != subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT):
            raise ValueError("發布來源不是指定commit：" + name)
        hashes[name] = sha(ROOT / name)
    if (
        hashes[POLICY_PATH] != POLICY_SHA
        or hashes["docs/lesson-products/data-v2-manifest.json"] != spec["data_manifest_sha256"]
    ):
        raise ValueError("發布資料或政策不是已固定版本")
    return spec, hashes


def register(archive):
    import modal

    app = modal.App("tiny-perceptron-lesson-products-release")
    volume = modal.Volume.from_name("tiny-perceptron-course", create_if_missing=False)
    secret = modal.Secret.from_name(os.environ.get("HF_MODAL_SECRET") or "codex_cloud", required_keys=["HF_TOKEN"])
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install("huggingface-hub==1.33.0")
        .env({"PYTHONPATH": "/repo", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        .add_local_dir(ROOT / "scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
        .add_local_dir(ROOT / "docs/lesson-products", "/repo/docs/lesson-products")
        .add_local_file(ROOT / POLICY_PATH, "/repo/" + POLICY_PATH)
        .add_local_file(
            ROOT / ".github/workflows/lesson-products-publish.yml",
            "/repo/.github/workflows/lesson-products-publish.yml",
        )
        .add_local_file(archive, "/release.tar.gz")
    )

    @app.function(
        serialized=True,
        image=image,
        cpu=(1, 1),
        memory=(2048, 2048),
        timeout=600,
        retries=0,
        max_containers=1,
        scaledown_window=2,
        volumes={"/course": volume},
        secrets=[secret],
    )
    def publish_remote(run_id, revision, spec, hashes, old_ledger, old_quota, repo):
        import modal
        from huggingface_hub import HfApi, hf_hub_download

        check_source_files("/repo", hashes)
        volume.reload()
        if sha(LEDGER) != old_ledger or sha(QUOTA) != old_quota:
            raise ValueError("共享帳本已變，拒絕發布預約；不重設")
        ledger = json.loads(LEDGER.read_bytes())
        quota = json.loads(QUOTA.read_bytes())
        policy = json.loads((Path("/repo") / POLICY_PATH).read_bytes())
        live = live_snapshot()
        live["gross_quota_snapshot"] = official_snapshot(modal.Workspace.from_context(), policy, POLICY_SHA)
        entry, following = reserve_entry(
            ledger,
            run_id,
            "lesson-products-release-v1",
            revision,
            spec["transport_sha256"],
            spec,
            canonical_sha(spec),
            live,
            hashes,
            policy,
            POLICY_SHA,
            quota,
        )
        entry.update(
            experiment_id="selftrained_lesson_products_release",
            status="running",
            function_call_id=modal.current_function_call_id(),
        )
        write_json(QUOTA, following)
        write_json(LEDGER, ledger)
        volume.commit()
        directory = Path("/course/lesson-products-publications") / run_id
        directory.mkdir(parents=True, exist_ok=True)
        receipt = {
            "run_id": run_id,
            "revision": revision,
            "source_hashes": hashes,
            "status": "failed",
            "gpu_used": False,
            "reserved_usd": entry["reserved_usd"],
            "gross_quota_guard": entry["gross_quota_guard"],
        }
        try:
            payload = unpack_verified("/release.tar.gz", spec["transport_sha256"], directory / "payload")
            receipt["verified_payload"] = verify_release(payload, spec)
            api = HfApi(token=os.environ["HF_TOKEN"])
            info = api.repo_info(repo, repo_type="model")
            if info.private:
                raise ValueError("正式成品需匿名可讀的既有public repo")
            existing = {
                name for name in api.list_repo_files(repo, revision=info.sha) if name.startswith(spec["prefix"] + "/")
            }
            if existing:
                if existing != {f"{spec['prefix']}/{name}" for name in spec["files"]}:
                    raise ValueError("prefix已有部分或其他內容，不覆寫；請另立發布版本")
                public_revision = info.sha
                receipt["recovered_existing_prefix"] = True
            else:
                commit = api.upload_folder(
                    repo_id=repo,
                    folder_path=str(payload),
                    path_in_repo=spec["prefix"],
                    commit_message=f"Publish frozen B/C lesson products: {run_id}",
                )
                public_revision = commit.oid
            receipt["public"] = {"repo": repo, "revision": public_revision, "prefix": spec["prefix"]}
            verified = []
            for name, expected in spec["files"].items():
                path = Path(
                    hf_hub_download(
                        repo, f"{spec['prefix']}/{name}", revision=public_revision, token=False, force_download=True
                    )
                )
                if sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                    raise ValueError("匿名固定commit下載不同：" + name)
                verified.append({"path": name, **expected})
            receipt.update(status="completed", anonymous_downloads=verified)
        except Exception as error:
            receipt.update(error_type=type(error).__name__, error=redact_error(error))
        write_json(directory / "execution.json", receipt)
        ledger = json.loads(LEDGER.read_bytes())
        finish_entry(ledger, run_id, receipt["status"], receipt)
        write_json(LEDGER, ledger)
        volume.commit()
        return receipt

    @app.function(
        serialized=True,
        image=image,
        cpu=(0.25, 0.25),
        memory=(1024, 1024),
        timeout=180,
        retries=0,
        max_containers=1,
        scaledown_window=2,
        volumes={"/course": volume},
    )
    def finish_remote(run_id, call_id):
        import modal

        volume.reload()
        ledger = json.loads(LEDGER.read_bytes())
        entry = next((e for e in ledger["reservations"] if e["run_id"] == run_id), None)
        if entry is None:
            return {"status": "not_reserved"}
        if entry.get("experiment_id") != "selftrained_lesson_products_release":
            raise ValueError("不是本發布器預約")
        if entry["status"] not in ("completed", "failed", "failed-or-cancelled"):
            actual = entry.get("function_call_id")
            if not actual or (call_id and actual != call_id):
                raise ValueError("缺少實際CPU call終止身分；保留非terminal預約")
            confirm_stopped(
                modal.FunctionCall.from_id(actual),
                (
                    modal.exception.FunctionTimeoutError,
                    modal.exception.RemoteError,
                    modal.exception.OutputExpiredError,
                    ValueError,
                ),
            )
            volume.reload()
            ledger = json.loads(LEDGER.read_bytes())
            entry = next(e for e in ledger["reservations"] if e["run_id"] == run_id)
            if entry["status"] not in ("completed", "failed", "failed-or-cancelled"):
                finish_entry(ledger, run_id, "failed-or-cancelled")
                write_json(LEDGER, ledger)
                volume.commit()
        return {"status": entry["status"], "bound_retained": True}

    return app, publish_remote, finish_remote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs/lesson-products-release/transport.tar.gz")
    parser.add_argument("--release-repo", default=os.environ.get("HF_RELEASE_REPO"))
    options = parser.parse_args()
    safe_name(options.run_id)
    spec, hashes = inputs(options.revision, options.spec)
    options.bundle.parent.mkdir(parents=True, exist_ok=True)
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo != "birdhackor/tiny-perceptron-vlm":
        raise ValueError("來源需同一repo的固定release asset")
    parent = json.loads(subprocess.check_output(["gh", "api", f"repos/{repo}/releases/{spec['release_id']}"]))
    asset = next((item for item in parent["assets"] if item["id"] == spec["asset_id"]), None)
    if (
        parent.get("draft") is not True
        or parent.get("target_commitish") != spec["release_commit"]
        or asset is None
        or asset.get("name") != spec["asset_name"]
        or asset.get("size") != spec["asset_bytes"]
    ):
        raise ValueError("GitHub draft parent／commit／asset metadata不符已提交來源")
    if not options.bundle.exists():
        with options.bundle.open("wb") as stream:
            subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{repo}/releases/assets/{spec['asset_id']}",
                    "-H",
                    "Accept: application/octet-stream",
                ],
                stdout=stream,
                check=True,
            )
    if sha(options.bundle) != spec["transport_sha256"]:
        raise ValueError("GitHub transport asset SHA不同")
    if options.bundle.stat().st_size != spec["asset_bytes"]:
        raise ValueError("GitHub transport大小不同")
    preflight = unpack_verified(
        options.bundle, spec["transport_sha256"], options.bundle.parent / ("preflight-" + options.run_id)
    )
    verify_release(preflight, spec)
    if not options.release_repo or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", options.release_repo):
        raise ValueError("需同一既有公開HF model repo")
    policy = json.loads((ROOT / POLICY_PATH).read_bytes())
    live = read_live_ledger(policy, POLICY_SHA)
    reserve_entry(
        json.loads(json.dumps(live["ledger"])),
        options.run_id,
        "lesson-products-release-v1",
        options.revision,
        spec["transport_sha256"],
        spec,
        canonical_sha(spec),
        live["live"],
        hashes,
        policy,
        POLICY_SHA,
        live["gross_quota_state"],
    )
    app, publish, finish = register(options.bundle)
    output = ROOT / "outputs/lesson-products-release/result.json"
    result = {"status": "failed", "run_id": options.run_id}
    with app.run():
        call = None
        try:
            call = publish.spawn(
                options.run_id,
                options.revision,
                spec,
                hashes,
                live["ledger_sha256"],
                live["gross_quota_state_sha256"],
                options.release_repo,
            )
            result["function_call_id"] = call.object_id
            write_json(output, result)
            result["execution"] = call.get()
            result["status"] = result["execution"]["status"]
        except Exception as error:
            result.update(error_type=type(error).__name__, error=redact_error(error))
        finally:
            try:
                latest = read_live_ledger(policy, POLICY_SHA)
                entry = next((e for e in latest["ledger"]["reservations"] if e["run_id"] == options.run_id), None)
                if entry and entry["status"] not in ("completed", "failed", "failed-or-cancelled"):
                    result["finish"] = finish.remote(options.run_id, call.object_id if call else None)
                result["billing_after_snapshot"] = latest["live"]["gross_quota_snapshot"]
            except Exception as error:
                result["finish_error"] = {"type": type(error).__name__, "message": redact_error(error)}
                result["status"] = "failed"
            result["all_bounds_retained"] = True
            write_json(output, result)
    print(json.dumps({"status": result["status"], "run_id": options.run_id, "gpu_used": False}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
