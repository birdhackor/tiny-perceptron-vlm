"""沿用已修正gross額度守門；新成品不呼叫舊課程訓練器。"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from scripts.lesson_products.checkpoints import FILES as CHECKPOINT_FILES
from scripts.lesson_products.checkpoints import bundle, prepare_resume, validate_resume
from scripts.selftrained.finance import official_snapshot
from scripts.selftrained.modal_runner import (
    canonical_sha,
    finish_entry,
    live_snapshot,
    read_live_ledger,
    reserve_entry,
    safe_name,
    sha256,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = "docs/selftrained/v2-gross-quota-20261006.json"
POLICY_SHA = "557b2bd1135c41beff0fdc38bc69ed8a4c36ddc06d14b917d16bf5b732867372"
DATA_MANIFEST = "docs/lesson-products/data-v2-manifest.json"
LEDGER = Path("/course/budget.json")
QUOTA = Path("/course/gross-quota.json")
SOURCE_FILES = [
    "tiny_perceptron/lesson_products/data.py",
    "tiny_perceptron/lesson_products/model.py",
    "tiny_perceptron/lesson_products/training.py",
    "tiny_perceptron/lesson_products/train.py",
    "scripts/lesson_products/modal_runner.py",
    "scripts/lesson_products/checkpoints.py",
    "scripts/selftrained/finance.py",
    "scripts/selftrained/modal_runner.py",
    "uv.lock",
    "pyproject.toml",
    ".github/workflows/gpu-smoke.yml",
    POLICY_PATH,
    DATA_MANIFEST,
]


def validate_job(job):
    expected = {
        "version",
        "stage",
        "job_kind",
        "product",
        "mode",
        "max_steps",
        "train_seconds",
        "wall_seconds",
        "batch_size",
        "seed",
        "learning_rate",
        "depth_weights",
        "gross_quota_policy",
        "resume",
    }
    if (
        set(job) != expected
        or job["version"] != 1
        or job["stage"] != "pretrain"
        or job["job_kind"] != "lesson_product_train"
    ):
        raise ValueError("需使用固定欄位的成品訓練描述，不是舊selftrained訓練流程")
    modes = {"b": ("multi", "fixed3", "odd"), "c": ("candidate",)}
    if job["mode"] not in modes.get(job["product"], ()):
        raise ValueError("未知成品或訓練目標")
    for key in ("max_steps", "train_seconds", "wall_seconds", "batch_size", "seed"):
        if type(job[key]) is not int:
            raise ValueError("訓練規模需明確整數")
    if job["wall_seconds"] not in (600, 900, 1800) or not 1 <= job["train_seconds"] <= min(
        1500, job["wall_seconds"] - 300
    ):
        raise ValueError("訓練必須預留驗證、checkpoint與上傳時間")
    if not 1 <= job["max_steps"] <= 50000 or not 1 <= job["batch_size"] <= 256 or not 0 <= job["seed"] < 2**31:
        raise ValueError("超出有界規模")
    if type(job["learning_rate"]) not in (int, float) or not 0 < job["learning_rate"] <= 0.01:
        raise ValueError("不合法learning_rate")
    weights = job["depth_weights"]
    if (
        not isinstance(weights, list)
        or len(weights) != 5
        or any(type(x) not in (int, float) or not 0 < x <= 100 for x in weights)
    ):
        raise ValueError("需五個正的多輪loss權重")
    if job["gross_quota_policy"] != {"path": POLICY_PATH, "sha256": POLICY_SHA}:
        raise ValueError("不能更換或提高已授權gross額度")
    validate_resume(job["resume"])
    return job


def committed_inputs(revision, job_path):
    if (
        not re.fullmatch(r"[a-f0-9]{40}", revision)
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip() != revision
    ):
        raise ValueError("需要實際checkout的精確commit")
    path = Path(job_path)
    if path.parent.as_posix() != "docs/lesson-products/jobs" or path.suffix != ".json" or ".." in path.parts:
        raise ValueError("job需為已提交的版本化描述")
    hashes = {}
    for name in SOURCE_FILES + [job_path]:
        raw = (ROOT / name).read_bytes()
        if raw != subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT):
            raise ValueError("執行來源不符合指定commit：" + name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    if hashes[POLICY_PATH] != POLICY_SHA:
        raise ValueError("額度政策不符合預先核准版本")
    return validate_job(json.loads((ROOT / job_path).read_bytes())), hashes


def check_source_files(root, hashes):
    if any(sha256(Path(root) / name) != digest for name, digest in hashes.items()):
        raise ValueError("遠端程式或描述的SHA不同")


def hf_roundtrip(checkpoint_repo, release_repo, run_id, revision, job_sha):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=os.environ["HF_TOKEN"])
    if (
        not api.repo_info(checkpoint_repo, repo_type="model").private
        or api.repo_info(release_repo, repo_type="model").private
    ):
        raise ValueError("中途checkpoint需private、成品repo需public")
    raw = json.dumps(
        {
            "run_id": run_id,
            "code_revision": revision,
            "job_sha256": job_sha,
            "scope": "Capability preflight only; not model weights or training success.",
        },
        sort_keys=True,
    ).encode()
    receipts = []
    for repo, public in ((checkpoint_repo, False), (release_repo, True)):
        name = f"lesson-products/preflight/{run_id}.json"
        commit = api.upload_file(
            repo_id=repo,
            path_or_fileobj=raw,
            path_in_repo=name,
            commit_message=f"Verify lesson product upload: {run_id}",
        )
        downloaded = hf_hub_download(
            repo, name, revision=commit.oid, token=False if public else os.environ["HF_TOKEN"], force_download=True
        )
        if Path(downloaded).read_bytes() != raw:
            raise ValueError("HF寫入後固定commit下載指紋不符")
        receipts.append(
            {
                "repo": repo,
                "revision": commit.oid,
                "path": name,
                "anonymous": public,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return receipts


def confirm_stopped(call, terminal_errors):
    call.cancel(terminate_containers=True)
    try:
        call.get(timeout=30)
    except terminal_errors:
        return


def register(action, data_directory):
    import modal

    app = modal.App("tiny-perceptron-lesson-products")
    volume = modal.Volume.from_name("tiny-perceptron-course", create_if_missing=False)
    secret = modal.Secret.from_name(os.environ.get("HF_MODAL_SECRET") or "codex_cloud", required_keys=["HF_TOKEN"])

    def sources(image):
        return (
            image.env(
                {"PYTHONPATH": "/repo", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
            )
            .workdir("/repo")
            .add_local_dir(ROOT / "tiny_perceptron", "/repo/tiny_perceptron", ignore=["**/__pycache__/**"])
            .add_local_dir(ROOT / "scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
            .add_local_dir(ROOT / "docs/lesson-products", "/repo/docs/lesson-products")
            .add_local_file(ROOT / POLICY_PATH, "/repo/" + POLICY_PATH)
            .add_local_file(ROOT / "uv.lock", "/repo/uv.lock")
            .add_local_file(ROOT / "pyproject.toml", "/repo/pyproject.toml")
            .add_local_file(ROOT / ".github/workflows/gpu-smoke.yml", "/repo/.github/workflows/gpu-smoke.yml")
        )

    control = sources(modal.Image.debian_slim(python_version="3.13").pip_install("huggingface-hub==1.33.0"))
    image = control
    if action == "run":
        image = sources(
            modal.Image.debian_slim(python_version="3.13").uv_sync(
                str(ROOT), extras=["cu126", "selftrained"], uv_version="0.12.22", extra_options="--no-dev"
            )
        ).add_local_dir(Path(data_directory), "/input-data")

    @app.function(
        serialized=True,
        image=control,
        cpu=(0.25, 0.25),
        memory=(1024, 1024),
        volumes={"/course": volume},
        secrets=[secret],
        timeout=180,
        retries=0,
        max_containers=1,
        scaledown_window=2,
    )
    def reserve_remote(
        run_id, revision, job, hashes, previous_ledger_sha, previous_quota_sha, checkpoint_repo, release_repo
    ):
        import modal

        check_source_files("/repo", hashes)
        validate_job(job)
        volume.reload()
        if (
            not LEDGER.is_file()
            or sha256(LEDGER) != previous_ledger_sha
            or not QUOTA.is_file()
            or sha256(QUOTA) != previous_quota_sha
        ):
            raise RuntimeError("共享帳本在preflight後有變，拒絕預約；不可重設")
        ledger = json.loads(LEDGER.read_bytes())
        quota = json.loads(QUOTA.read_bytes())
        policy = json.loads((Path("/repo") / POLICY_PATH).read_bytes())
        live = live_snapshot()
        live["gross_quota_snapshot"] = official_snapshot(modal.Workspace.from_context(), policy, POLICY_SHA)
        entry, following = reserve_entry(
            ledger,
            run_id,
            "lesson-products-v2",
            revision,
            hashes[DATA_MANIFEST],
            job,
            canonical_sha(job),
            live,
            hashes,
            policy,
            POLICY_SHA,
            quota,
        )
        entry["experiment_id"] = "selftrained_lesson_products"
        entry["product"] = job["product"]
        write_json(QUOTA, following)
        write_json(LEDGER, ledger)
        volume.commit()
        directory = Path("/course/lesson-products") / run_id
        directory.mkdir(parents=True, exist_ok=True)
        receipt = {
            "run_id": run_id,
            "revision": revision,
            "job": job,
            "source_hashes": hashes,
            "reserved_usd": entry["reserved_usd"],
            "gross_quota_guard": entry["gross_quota_guard"],
        }
        try:
            if job["resume"] and job["resume"]["run_id"] == run_id:
                raise ValueError("恢復工作必須另立attempt預約，不能覆寫原run")
            receipt["resume_preflight"] = prepare_resume(
                job["resume"], directory / "resume", "/course", checkpoint_repo, hashes, job, ledger
            )
            receipt["hf_preflight"] = hf_roundtrip(checkpoint_repo, release_repo, run_id, revision, canonical_sha(job))
            write_json(directory / "reservation.json", receipt)
            volume.commit()
            return receipt
        except Exception:
            finish_entry(ledger, run_id, "failed")
            write_json(LEDGER, ledger)
            volume.commit()
            raise

    @app.function(
        serialized=True,
        image=control,
        cpu=(0.25, 0.25),
        memory=(1024, 1024),
        volumes={"/course": volume},
        timeout=180,
        retries=0,
        max_containers=1,
        scaledown_window=2,
    )
    def finish_remote(run_id, call_id=None, no_gpu_dispatch=False):
        import modal

        volume.reload()
        ledger = json.loads(LEDGER.read_bytes())
        entry = next(x for x in ledger["reservations"] if x["run_id"] == run_id)
        if entry.get("experiment_id") != "selftrained_lesson_products":
            raise ValueError("不是本成品runner的預約")
        if entry["status"] not in ("completed", "failed", "failed-or-cancelled"):
            actual_call = entry.get("function_call_id") or call_id
            if entry.get("function_call_id") and call_id and call_id != entry["function_call_id"]:
                raise ValueError("call ID與實際GPU紀錄不符")
            if actual_call:
                call = modal.FunctionCall.from_id(actual_call)
                confirm_stopped(
                    call,
                    (
                        modal.exception.FunctionTimeoutError,
                        modal.exception.RemoteError,
                        modal.exception.OutputExpiredError,
                        ValueError,
                    ),
                )
                # 輪詢尚未完成的TimeoutError或網路錯誤會往上拋，不改terminal。
            elif entry["status"] != "reserved" or not no_gpu_dispatch:
                raise RuntimeError("沒有GPU終止證據，保持非terminal狀態與原bound")
            volume.reload()
            ledger = json.loads(LEDGER.read_bytes())
            entry = next(x for x in ledger["reservations"] if x["run_id"] == run_id)
            if entry["status"] not in ("completed", "failed", "failed-or-cancelled"):
                finish_entry(ledger, run_id, "failed-or-cancelled")
                write_json(LEDGER, ledger)
                volume.commit()
        return {"run_id": run_id, "status": entry["status"], "bound_retained": True}

    @app.function(
        serialized=True,
        image=image,
        gpu="L4",
        cpu=(2, 2),
        memory=(8192, 8192),
        volumes={"/course": volume},
        secrets=[secret],
        timeout=1800,
        retries=0,
        max_containers=1,
        scaledown_window=2,
    )
    def execute_remote(run_id, revision, job, hashes, checkpoint_repo):
        import time

        import modal
        import torch
        from huggingface_hub import HfApi

        from tiny_perceptron.lesson_products.train import train

        started = time.monotonic()
        check_source_files("/repo", hashes)
        validate_job(job)
        volume.reload()
        ledger = json.loads(LEDGER.read_bytes())
        entry = next(x for x in ledger["reservations"] if x["run_id"] == run_id)
        if (
            entry["status"] != "reserved"
            or entry["job_sha256"] != canonical_sha(job)
            or entry["source_sha256"] != hashes
        ):
            raise ValueError("沒有此commit/job/source的有效預約")
        directory = Path("/course/lesson-products") / run_id
        if not (directory / "reservation.json").is_file():
            raise ValueError("HF公開寫入／匿名下載preflight尚未完成")
        if sha256(Path("/input-data/manifest.json")) != hashes[DATA_MANIFEST]:
            raise ValueError("實際資料manifest與提交描述不符")
        entry["status"] = "running"
        entry["function_call_id"] = modal.current_function_call_id()
        write_json(LEDGER, ledger)
        volume.commit()
        receipt = {
            "run_id": run_id,
            "code_revision": revision,
            "job": job,
            "source_hashes": hashes,
            "status": "failed",
            "reserved_usd": entry["reserved_usd"],
            "gross_quota_guard": entry["gross_quota_guard"],
        }
        try:
            if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0):
                raise RuntimeError("需要實際L4，不能用CPU假稱GPU訓練")
            receipt["gpu_name"] = torch.cuda.get_device_name(0)
            api = HfApi(token=os.environ["HF_TOKEN"])

            def persist(path, current):
                bundle(directory, run_id, current, hashes, job)
                volume.commit()
                if current % 1000 == 0:
                    backup = api.upload_folder(
                        repo_id=checkpoint_repo,
                        folder_path=str(directory),
                        path_in_repo=f"lesson-products/{run_id}",
                        allow_patterns=[*CHECKPOINT_FILES, "checkpoint-bundle.json"],
                        commit_message=f"Save paired resumable lesson step {current}: {run_id}",
                    )
                    write_json(
                        directory / "checkpoint-backup.json",
                        {
                            "revision": backup.oid,
                            "step": current,
                            "bundle_sha256": sha256(directory / "checkpoint-bundle.json"),
                        },
                    )
                    volume.commit()

            training = train(
                "/input-data",
                directory,
                job["product"],
                job["mode"],
                max_steps=job["max_steps"],
                max_seconds=min(job["train_seconds"], int(job["wall_seconds"] - 240 - (time.monotonic() - started))),
                batch_size=job["batch_size"],
                learning_rate=job["learning_rate"],
                seed=job["seed"],
                device="cuda",
                depth_weights=job["depth_weights"],
                on_checkpoint=persist,
                resume=directory / "resume/last.pt" if job["resume"] else None,
                resume_best=directory / "resume/best.pt" if job["resume"] else None,
            )
            # 最後一個step也有同一組best/last及progress，不依賴200步邊界。
            progress = (
                json.loads((directory / "progress.json").read_bytes()) if (directory / "progress.json").exists() else {}
            )
            progress["final_training_receipt"] = training
            write_json(directory / "progress.json", progress)
            bundle(directory, run_id, training["last_step"], hashes, job)
            volume.commit()
            commit = api.upload_folder(
                repo_id=checkpoint_repo,
                folder_path=str(directory),
                path_in_repo=f"lesson-products/{run_id}",
                allow_patterns=[
                    "best.pt",
                    "last.pt",
                    "model.pt",
                    "config.json",
                    "tokenizer.json",
                    "training-receipt.json",
                    "progress.json",
                    "reservation.json",
                    "checkpoint-bundle.json",
                    "checkpoint-backup.json",
                ],
                commit_message=f"Save independently trained lesson product: {run_id}",
            )
            receipt.update(
                status="completed",
                training=training,
                private_checkpoint={
                    "repo": checkpoint_repo,
                    "revision": commit.oid,
                    "prefix": f"lesson-products/{run_id}",
                    "bundle_sha256": sha256(directory / "checkpoint-bundle.json"),
                },
            )
        except Exception as error:
            receipt.update(error_type=type(error).__name__, error=str(error)[:1500])
        receipt["remote_wall_seconds"] = time.monotonic() - started
        write_json(directory / "execution.json", receipt)
        # 結束狀態不釋回bounds；gross sidecar與所有舊預約保持。
        ledger = json.loads(LEDGER.read_bytes())
        finish_entry(ledger, run_id, receipt["status"], receipt)
        write_json(LEDGER, ledger)
        volume.commit()
        return receipt

    return app, reserve_remote, execute_remote, finish_remote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "ledger", "run", "finish"), required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--call-id", help="實際未完成工作的取消收尾call ID")
    parser.add_argument("--data-directory", type=Path, default=ROOT / "outputs/lesson-products/data-v2")
    parser.add_argument("--checkpoint-repo", default=os.environ.get("HF_CHECKPOINT_REPO"))
    parser.add_argument("--release-repo", default=os.environ.get("HF_RELEASE_REPO"))
    options = parser.parse_args()
    safe_name(options.run_id)
    job, hashes = committed_inputs(options.revision, options.job)
    output = ROOT / "outputs/lesson-products/runner-result.json"
    policy = json.loads((ROOT / POLICY_PATH).read_bytes())
    if options.action == "prepare":
        from tiny_perceptron.lesson_products.data import write_dataset

        manifest = json.loads((ROOT / DATA_MANIFEST).read_bytes())
        config = manifest["config"]
        write_dataset(
            options.data_directory,
            per_length=config["per_length"],
            stress_per_length=config["stress_per_length"],
            services=config["services"],
        )
        if sha256(options.data_directory / "manifest.json") != hashes[DATA_MANIFEST]:
            raise ValueError("重建資料與事先提交manifest不同")
        print(json.dumps({"data_prepared": True, "data_manifest_sha256": hashes[DATA_MANIFEST]}))
        return
    live = read_live_ledger(policy, POLICY_SHA)
    if options.action == "ledger":
        write_json(output, live)
        print(
            json.dumps({"gross_quota_snapshot": live["live"]["gross_quota_snapshot"], "legacy_totals_audit_only": True})
        )
        return
    if options.action == "run":
        if not options.checkpoint_repo or not options.release_repo:
            raise ValueError("需要已核准的private checkpoint/public release repo")
        if sha256(options.data_directory / "manifest.json") != hashes[DATA_MANIFEST]:
            raise ValueError("先用prepare重建固定資料，不可臨時換資料")
        probe_ledger = json.loads(json.dumps(live["ledger"]))
        reserve_entry(
            probe_ledger,
            options.run_id,
            "lesson-products-v2",
            options.revision,
            hashes[DATA_MANIFEST],
            job,
            canonical_sha(job),
            live["live"],
            hashes,
            policy,
            POLICY_SHA,
            live["gross_quota_state"],
        )
    app, reserve, execute, finish = register(options.action, options.data_directory)
    result = {
        "run_id": options.run_id,
        "code_revision": options.revision,
        "status": "failed",
        "gpu_execution_status": "not_dispatched",
    }
    with app.run():
        if options.action == "finish":
            write_json(output, finish.remote(options.run_id, options.call_id))
            return
        call = None
        dispatch_attempted = False
        try:
            result["reservation"] = reserve.remote(
                options.run_id,
                options.revision,
                job,
                hashes,
                live["ledger_sha256"],
                live["gross_quota_state_sha256"],
                options.checkpoint_repo,
                options.release_repo,
            )
            dispatch_attempted = True
            result["gpu_execution_status"] = "dispatch_attempted"
            call = execute.with_options(timeout=job["wall_seconds"]).spawn(
                options.run_id, options.revision, job, hashes, options.checkpoint_repo
            )
            result["function_call_id"] = call.object_id
            result["gpu_dispatch_submitted"] = True
            result["gpu_execution_status"] = "submitted"
            write_json(output, result)
            result["execution"] = call.get()
            result["gpu_execution_status"] = (
                "confirmed" if result["execution"].get("gpu_name") else "result_received_identity_unverified"
            )
            result["status"] = result["execution"]["status"]
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error)[:1500])
        finally:
            # 客戶端/遠端timeout時只有這一個有界CPU收尾，預約+收尾不超過兩個control。
            try:
                latest = read_live_ledger(policy, POLICY_SHA)
                entry = next((x for x in latest["ledger"]["reservations"] if x["run_id"] == options.run_id), None)
                if entry and entry["status"] not in ("completed", "failed", "failed-or-cancelled"):
                    result["finish"] = finish.remote(
                        options.run_id, call.object_id if call else None, not dispatch_attempted
                    )
                result["billing_after_snapshot"] = latest["live"]["gross_quota_snapshot"]
            except Exception as error:
                result["finish_error"] = {"type": type(error).__name__, "message": str(error)[:1500]}
                result["status"] = "failed"
            result["all_bounds_retained"] = True
            write_json(output, result)
    print(json.dumps({"status": result["status"], "run_id": options.run_id, "all_bounds_retained": True}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
