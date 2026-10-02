"""逐項執行課程實驗；GPU 有界、私有備份，審閱後才另行發布推論檔。"""

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME_ROOT = Path("/course")
BUDGET_USD = Decimal("10.00")
GPU_RESERVATION_USD = Decimal("0.22")
CPU_RESERVATION_USD = Decimal("0.04")
REFERENCE_RATES = {"L4_per_second": "0.000222", "physical_cpu_per_second": "0.0000131", "GiB_per_second": "0.00000222"}
app = modal.App("tiny-perceptron-course-experiments")
volume = modal.Volume.from_name("tiny-perceptron-course", create_if_missing=True)
hf_secret = modal.Secret.from_name(os.environ.get("HF_MODAL_SECRET") or "codex_cloud", required_keys=["HF_TOKEN"])
cpu_image = modal.Image.debian_slim(python_version="3.13").pip_install("huggingface-hub==1.33.0")


def asset_ignore(path):
    """只上傳 Actions 已選定的 archive；不把 LFS pointer 當訓練資料。"""
    path = Path(path)
    if path.suffixes[-2:] != [".tar", ".gz"]:
        return False
    selected = {Path(name).name for name in os.environ.get("COURSE_ASSET_PATHS", "").splitlines() if name}
    if path.name not in selected:
        return True
    local_path = path if path.is_absolute() else ROOT / "assets/training" / path
    with local_path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"選定的資產還是 LFS pointer：{path.name}")
    return False


gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(str(ROOT), extras=["cu126"], uv_version="0.12.22", extra_options="--no-dev")
    .env({"PYTHONPATH": "/app", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .workdir("/app")
    .add_local_dir(ROOT / "tiny_perceptron", "/app/tiny_perceptron", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "scripts", "/app/scripts", ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / "docs/course-experiments/plan.json", "/app/docs/course-experiments/plan.json")
    .add_local_dir(ROOT / "assets/training", "/app/assets/training", ignore=asset_ignore)
)


def json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value) or ".." in value:
        raise ValueError("名稱只能含英數字、底線、連字號與單個點，不能含路徑")
    return value


def billing_snapshot():
    """在 Actions CPU 讀取；不把 Modal 登入憑證注入訓練 container。"""
    billing = modal.Workspace.from_context().billing
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "scope": "account-wide workspace billing; not attributable to this project; updates can lag",
        "summary": json_value(asdict(billing.summary())),
        "rates": {key: str(value) for key, value in billing.rates().items()},
        "reference_rates": REFERENCE_RATES,
    }


def compute_reservation_guard(rates):
    """以 live 單價核對保留額；不認識的 resource key 先停止，避免猜單位。"""
    aliases = {
        "gpu": {"l4", "gpul4"},
        "cpu": {"cpu", "physicalcpu", "cpuphysical", "cpuphysicalcore", "physicalcpucore"},
        "memory": {"memory", "memorygib", "gib", "ramgib"},
    }
    matched = {}
    for key, value in rates.items():
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        for resource, names in aliases.items():
            if normalized in names:
                matched[resource] = Decimal(value)
    if set(matched) != set(aliases):
        raise ValueError("無法辨認 live billing rates 的 L4/physical CPU/GiB 單位；先檢查 preflight 的 rates")
    # 600 秒函式、2 秒 idle，另保留 US$0.04 給 CPU preflight/備份/帳本。
    gpu_envelope = Decimal("602") * (matched["gpu"] + 2 * matched["cpu"] + 8 * matched["memory"])
    estimate = gpu_envelope + CPU_RESERVATION_USD
    if estimate > GPU_RESERVATION_USD:
        raise RuntimeError(f"live 單價的保守估算 US${estimate} 超過本次 US${GPU_RESERVATION_USD} 保留額；未啟動 GPU")
    return {
        "gpu_envelope_usd": str(gpu_envelope),
        "cpu_auxiliary_allowance_usd": str(CPU_RESERVATION_USD),
        "conservative_compute_usd": str(estimate),
        "units": "L4 second; physical CPU core second; GiB second",
        "excludes": "storage and network; account-wide billing cannot isolate this project",
    }


def reserve_budget(ledger, run_id, batch_id, experiment_id, mode, snapshot):
    """保留失敗工作的最壞用量；每次嘗試都佔額，不靠延遲帳單回收。"""
    entries = ledger.setdefault("reservations", [])
    if any(entry["run_id"] == run_id for entry in entries):
        raise ValueError("這個 run_id 已保留預算；每次嘗試需使用新的 run_id")
    reservation = GPU_RESERVATION_USD if mode == "run" else CPU_RESERVATION_USD
    guard = compute_reservation_guard(snapshot["rates"]) if mode == "run" else None
    previous = sum((Decimal(entry["reserved_usd"]) for entry in entries), Decimal("0"))
    if previous + reservation > BUDGET_USD:
        raise RuntimeError(f"本任務累計保留額 {previous} + {reservation} 超過 US${BUDGET_USD}；未啟動 GPU")
    entries.append(
        {
            "run_id": run_id,
            "batch_id": batch_id,
            "experiment_id": experiment_id,
            "mode": mode,
            "reserved_usd": str(reservation),
            "status": "reserved",
            "billing_before": snapshot,
            "compute_guard": guard,
        }
    )
    ledger.update(budget_usd=str(BUDGET_USD), reserved_total_usd=str(previous + reservation))
    return {
        "budget_usd": str(BUDGET_USD),
        "reserved_total_usd": str(previous + reservation),
        "this_job_usd": str(reservation),
    }


@app.function(
    image=cpu_image,
    cpu=(0.25, 0.25),
    memory=(512, 512),
    secrets=[hf_secret],
    volumes={"/course": volume},
    timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def preflight(checkpoint_repo, release_repo, run_id, batch_id, experiment_id, mode, revision, snapshot):
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    if not api.repo_info(checkpoint_repo, repo_type="model").private:
        raise RuntimeError("checkpoint repo 必須是 Private")
    if api.repo_info(release_repo, repo_type="model").private:
        raise RuntimeError("release repo 必須是 Public")
    volume.reload()
    ledger_path = VOLUME_ROOT / "budget.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    budget = reserve_budget(ledger, run_id, batch_id, experiment_id, mode, snapshot)
    write_json(ledger_path, ledger)
    volume.commit()
    proof = {"run_id": run_id, "revision": revision, "purpose": "GPU course upload permission check; no model or data"}
    commits = {}
    for label, repo in (("checkpoint", checkpoint_repo), ("release", release_repo)):
        commit = api.upload_file(
            repo_id=repo,
            path_or_fileobj=json.dumps(proof).encode(),
            path_in_repo=f"course-infra/{batch_id}/{run_id}.json",
            commit_message=f"Verify course {label} upload: {run_id}",
        )
        commits[label] = commit.oid
    return {
        "checkpoint_private": True,
        "release_public": True,
        "hf_read_and_write": True,
        "permission_commits": commits,
        "budget": budget,
    }


@app.function(
    image=gpu_image,
    gpu="L4",
    cpu=(2, 2),
    memory=(8192, 8192),
    volumes={"/course": volume},
    timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def train(experiment_id, batch_id, run_id, revision):
    from scripts.course_experiments.run import execute

    volume.reload()
    ledger = json.loads((VOLUME_ROOT / "budget.json").read_text())
    reservation = next(
        (
            entry
            for entry in ledger["reservations"]
            if entry["run_id"] == run_id
            and entry["mode"] == "run"
            and entry["batch_id"] == batch_id
            and entry["experiment_id"] == experiment_id
            and entry["status"] == "reserved"
        ),
        None,
    )
    if reservation is None:
        raise RuntimeError("GPU 工作必須先完成本任務預算保留")
    reservation["status"] = "running"
    write_json(VOLUME_ROOT / "budget.json", ledger)
    directory = VOLUME_ROOT / batch_id / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    for stale in ("result.json", "failure.json", "approved-public-exports.json"):
        (directory / stale).unlink(missing_ok=True)
    write_json(directory / "current-run.json", {"run_id": run_id, "revision": revision})
    volume.commit()
    finished = threading.Event()
    commit_errors = []

    def persist_periodically():
        # timeout 可能不進 finally，持續保存已原子換檔的 checkpoint。
        while not finished.wait(20):
            try:
                volume.commit()
            except Exception as error:
                commit_errors.append(type(error).__name__)

    persistence = threading.Thread(target=persist_periodically, daemon=True)
    persistence.start()
    started = time.perf_counter()
    try:
        result = execute(
            experiment_id,
            device="cuda",
            output=directory,
            dependencies=VOLUME_ROOT / batch_id,
            assets=Path("/app/assets/training"),
            revision=revision,
        )
        result["revision"] = revision
        result["status"] = "completed"
        result["modal"] = {
            "run_id": run_id,
            "batch_id": batch_id,
            "function_seconds": time.perf_counter() - started,
            "periodic_volume_commit_errors": commit_errors,
        }
        write_json(directory / "result.json", result)
        return json.dumps(result, ensure_ascii=False, allow_nan=False)
    except Exception as error:
        write_json(
            directory / "failure.json",
            {
                "run_id": run_id,
                "revision": revision,
                "experiment_id": experiment_id,
                "exception_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise
    finally:
        finished.set()
        persistence.join(timeout=5)
        volume.commit()


@app.function(
    image=cpu_image,
    cpu=(0.25, 0.25),
    memory=(1024, 1024),
    secrets=[hf_secret],
    volumes={"/course": volume},
    timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def backup(checkpoint_repo, experiment_id, batch_id, run_id, revision):
    from huggingface_hub import HfApi, hf_hub_download

    volume.reload()
    directory = VOLUME_ROOT / batch_id / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    result = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {
            "experiment_id": experiment_id,
            "revision": revision,
            "status": "failed-or-timed-out",
        }
    )
    if result.get("modal", {}).get("run_id") != run_id or result.get("revision") != revision:
        result = {
            "experiment_id": experiment_id,
            "revision": revision,
            "run_id": run_id,
            "status": "failed-or-timed-out",
        }
    prefix = f"course/{batch_id}/{experiment_id}/{run_id}"
    api = HfApi(token=os.environ["HF_TOKEN"])
    artifacts = [
        {"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink() and not path.name.endswith(".tmp")
    ]
    write_json(directory / "upload-manifest.json", {"artifacts": artifacts, "revision": revision, "run_id": run_id})
    commit = api.upload_folder(
        repo_id=checkpoint_repo,
        folder_path=str(directory),
        path_in_repo=prefix,
        ignore_patterns=["*.tmp", "**/*.tmp", "**/.cache/**"],
        commit_message=f"Save complete private course experiment {experiment_id}: {run_id}",
    )
    verified = []
    with tempfile.TemporaryDirectory(prefix="course-hf-check-") as temporary:
        for artifact in artifacts:
            if Path(artifact["path"]).suffix not in (".pt", ".pth", ".ckpt", ".safetensors"):
                continue
            downloaded = hf_hub_download(
                repo_id=checkpoint_repo,
                filename=f"{prefix}/{artifact['path']}",
                revision=commit.oid,
                token=os.environ["HF_TOKEN"],
                local_dir=temporary,
            )
            if sha256(downloaded) != artifact["sha256"]:
                raise RuntimeError(f"HF checkpoint 雜湊不一致：{artifact['path']}")
            verified.append(artifact["path"])
    result["hf"] = {
        "repo": checkpoint_repo,
        "revision": commit.oid,
        "prefix": prefix,
        "verified_checkpoints": verified,
        "private_full_training_state": True,
    }
    write_json(result_path, result)
    result_commit = api.upload_file(
        repo_id=checkpoint_repo,
        path_or_fileobj=str(result_path),
        path_in_repo=f"{prefix}/result.json",
        commit_message=f"Record course HF verification: {run_id}",
    )
    result["hf"]["result_revision"] = result_commit.oid
    write_json(result_path, result)
    volume.commit()
    return json.dumps(result, ensure_ascii=False, allow_nan=False)


def private_only_paths(value):
    paths = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "private_only_artifacts":
                for entry in item if isinstance(item, list) else [item]:
                    paths.append(entry if isinstance(entry, str) else entry["path"])
            else:
                paths.extend(private_only_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(private_only_paths(item))
    return paths


def approved_files(directory, approval, result):
    """發布清單需明示授權與精確雜湊；沒有清單就停止，避免猜測資料授權。"""
    if approval.get("approved") is not True or approval.get("revision") != result.get("revision"):
        raise ValueError("需要對此訓練 revision 的明確公開審閱清單")
    files = approval.get("files", [])
    if not files:
        raise ValueError("公開清單不能是空的")
    denied = private_only_paths(result.get("results", {}))
    for item in files:
        relative = Path(item["path"])
        path = directory / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
        ):
            raise ValueError("公開檔案路徑無效")
        if any(str(relative) == name or str(relative).startswith(str(name).rstrip("/") + "/") for name in denied):
            raise ValueError(f"禁止公開 private_only_artifacts：{relative}")
        if item.get("redistribution_approved") is not True or not item.get("license"):
            raise ValueError(f"缺少公開授權：{relative}")
        if relative.suffix in (".pt", ".pth", ".ckpt", ".safetensors") and item.get("kind") != "checkpoint":
            raise ValueError("權重檔必須經 checkpoint 推論匯出器，不能直接當 metadata 複製")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"審閱後檔案已變更：{relative}")
    return files


def inference_payload(saved, provenance):
    """只保留推論需要的欄位，不複製 optimizer、RNG、教師或未知資料欄位。"""
    allowed = (
        "format_version",
        "config",
        "model",
        "modal_config",
        "tokenizer",
        "bits",
        "task",
        "encoder",
    )
    clean = {key: saved[key] for key in allowed if key in saved}
    if "model" not in clean and "encoder" not in clean:
        raise ValueError("不是支援的推論 checkpoint，請另寫明確匯出器")
    clean["metadata"] = provenance
    return clean


@app.function(
    image=gpu_image,
    cpu=(1, 1),
    memory=(4096, 4096),
    secrets=[hf_secret],
    volumes={"/course": volume},
    timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def release(checkpoint_repo, release_repo, experiment_id, batch_id, run_id, approval_revision):
    import shutil

    import torch
    from huggingface_hub import HfApi, hf_hub_download

    volume.reload()
    directory = VOLUME_ROOT / batch_id / experiment_id
    result = json.loads((directory / "result.json").read_text())
    if result.get("status") != "completed":
        raise ValueError("未完成的實驗不能發布")
    approval_path = directory / "approved-public-exports.json"
    if approval_revision:
        approval_path = Path(
            hf_hub_download(
                repo_id=checkpoint_repo,
                token=os.environ["HF_TOKEN"],
                filename=f"course/{batch_id}/{experiment_id}/approved-public-exports.json",
                revision=approval_revision,
            )
        )
    if not approval_path.is_file():
        raise ValueError("尚未審閱：缺少 approved-public-exports.json；保留私有，不重訓")
    approval = json.loads(approval_path.read_text())
    files = approved_files(directory, approval, result)
    provenance = {
        "revision": result["revision"],
        "experiment_id": experiment_id,
        "batch_id": batch_id,
        "source_hf": result.get("hf", {}),
        "scope": "small educational model; use accompanying evaluation",
    }
    with tempfile.TemporaryDirectory(prefix="course-public-") as temporary:
        export = Path(temporary)
        manifest = []
        for item in files:
            source, destination = directory / item["path"], export / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if item.get("kind") == "checkpoint":
                torch.save(
                    inference_payload(torch.load(source, map_location="cpu", weights_only=True), provenance),
                    destination,
                )
            elif item.get("kind") in ("metadata", "dataset", "license"):
                shutil.copyfile(source, destination)
            else:
                raise ValueError("每個公開檔案需指定 checkpoint/metadata/dataset/license 類型")
            manifest.append(
                {
                    **item,
                    "source_sha256": item["sha256"],
                    "sha256": sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )
        write_json(export / "export-manifest.json", {"files": manifest, "provenance": provenance})
        (export / "README.md").write_text(
            "# Tiny Perceptron 教學實驗\n\n"
            f"實驗：`{experiment_id}`；程式版本：`{result['revision']}`。\n\n"
            "此目錄僅包含經審閱的教學推論檔，授權逐檔列於 export-manifest.json。"
            "量測範圍與評估限制請搭配教學閱讀；小型合成任務的結果不代表一般語言或多模態能力。\n",
            encoding="utf-8",
        )
        commit = HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
            repo_id=release_repo,
            folder_path=temporary,
            path_in_repo=f"course/{batch_id}/{experiment_id}",
            commit_message=f"Publish reviewed course inference files: {run_id}",
        )
    return json.dumps(
        {
            "experiment_id": experiment_id,
            "repo": release_repo,
            "revision": commit.oid,
            "files": manifest,
            "gpu_used": False,
        },
        ensure_ascii=False,
        allow_nan=False,
    )


@app.function(
    image=cpu_image,
    cpu=(0.25, 0.25),
    memory=(512, 512),
    volumes={"/course": volume},
    timeout=60,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def finish_budget(run_id, status, snapshot):
    volume.reload()
    path = VOLUME_ROOT / "budget.json"
    ledger = json.loads(path.read_text())
    entry = next(item for item in ledger["reservations"] if item["run_id"] == run_id)
    entry.update(status=status, billing_after=snapshot)
    before, after = entry["billing_before"]["summary"], snapshot["summary"]
    entry["workspace_metered_delta_usd"] = str(Decimal(after["metered_cost"]) - Decimal(before["metered_cost"]))
    entry["workspace_delta_scope"] = "account-wide; may include other projects and delayed prior usage"
    write_json(path, ledger)
    volume.commit()
    return {"budget_usd": ledger["budget_usd"], "reserved_total_usd": ledger["reserved_total_usd"], "entry": entry}


@app.local_entrypoint()
def main(
    experiment_id: str,
    checkpoint_repo: str,
    release_repo: str,
    run_id: str,
    revision: str,
    batch_id: str = "course-v1",
    mode: str = "run",
    approval_revision: str = "",
):
    for value in (experiment_id, batch_id, run_id):
        safe_name(value)
    if experiment_id == "preflight":
        mode = "preflight"
    if mode not in ("preflight", "run", "release"):
        raise ValueError("mode 必須是 preflight/run/release")
    for repo in (checkpoint_repo, release_repo):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("HF repo 必須是 owner/name")
    output = ROOT / "outputs/modal-course"
    before = billing_snapshot()
    write_json(output / "billing-before.json", before)
    access = preflight.remote(checkpoint_repo, release_repo, run_id, batch_id, experiment_id, mode, revision, before)
    result, error = {"experiment_id": experiment_id, "mode": mode, "preflight": access}, None
    try:
        if mode == "run":
            try:
                train.remote(experiment_id, batch_id, run_id, revision)
            finally:
                result = json.loads(backup.remote(checkpoint_repo, experiment_id, batch_id, run_id, revision))
                result["preflight"] = access
        elif mode == "release":
            result["release"] = json.loads(
                release.remote(checkpoint_repo, release_repo, experiment_id, batch_id, run_id, approval_revision)
            )
    except Exception as failure:
        error = failure
        result.update(status="failed", exception_type=type(failure).__name__, message=str(failure))
    finally:
        try:
            after = billing_snapshot()
            write_json(output / "billing-after.json", after)
            result["billing"] = finish_budget.remote(run_id, "failed" if error else "completed", after)
        except Exception as billing_error:
            result["billing_error"] = {"type": type(billing_error).__name__, "message": str(billing_error)}
            error = error or billing_error
        write_json(output / "result.json", result)
        print(json.dumps(json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
    if error:
        raise RuntimeError("課程實驗未完成；失敗用量仍保留，詳細資訊見 result.json") from error
