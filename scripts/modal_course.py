"""逐項執行課程實驗；GPU 有界、私有備份，審閱後才另行發布推論檔。"""

import hashlib
import json
import os
import re
import shutil
import subprocess
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
control_image = cpu_image.env({"PYTHONPATH": "/app"}).add_local_dir(
    ROOT / "scripts", "/app/scripts", ignore=["**/__pycache__/**"]
)


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
    .apt_install("build-essential")
    .run_commands(
        'python -c "import pathlib, sysconfig; '
        "p = pathlib.Path(sysconfig.get_path('include')) / 'Python.h'; "
        "assert p.is_file(), f'Missing matching Python development header: {p}'; print(p)\""
    )
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
    """使用實測 API 的每小時欄位；明確除以 3600，不由名稱猜單位。"""
    hourly_keys = {
        "gpu": "gpu_hour_cost_l4",
        "cpu": "cpu_hour_cost",
        "memory": "mem_gib_hour_cost",
    }
    if not all(key in rates for key in hourly_keys.values()):
        raise ValueError("live billing rates 缺少已確認每小時單位的 L4/physical CPU/GiB 欄位；先檢查 preflight")
    hourly = {resource: Decimal(rates[key]) for resource, key in hourly_keys.items()}
    if any(not value.is_finite() or value < 0 for value in hourly.values()):
        raise ValueError("live 每小時單價必須為有限的非負值")
    per_second = {resource: value / Decimal("3600") for resource, value in hourly.items()}
    # 600 秒函式、2 秒 idle，另保留 US$0.04 給 CPU preflight/備份/帳本。
    gpu_envelope = Decimal("602") * (per_second["gpu"] + 2 * per_second["cpu"] + 8 * per_second["memory"])
    estimate = gpu_envelope + CPU_RESERVATION_USD
    if estimate > GPU_RESERVATION_USD:
        raise RuntimeError(f"live 單價的保守估算 US${estimate} 超過本次 US${GPU_RESERVATION_USD} 保留額；未啟動 GPU")
    return {
        "gpu_envelope_usd": str(gpu_envelope),
        "cpu_auxiliary_allowance_usd": str(CPU_RESERVATION_USD),
        "conservative_compute_usd": str(estimate),
        "input_units": "USD per L4 hour; physical CPU core hour; GiB hour",
        "input_rates": {key: str(rates[key]) for key in hourly_keys.values()},
        "rates_per_second": {resource: str(value) for resource, value in per_second.items()},
        "seconds_per_hour": 3600,
        "excludes": "storage and network; account-wide billing cannot isolate this project",
    }


def reserve_budget(ledger, run_id, batch_id, experiment_id, mode, snapshot):
    """保留失敗工作的最壞用量；每次嘗試都佔額，不靠延遲帳單回收。"""
    entries = ledger.setdefault("reservations", [])
    if any(entry["run_id"] == run_id for entry in entries):
        raise ValueError("這個 run_id 已保留預算；每次嘗試需使用新的 run_id")
    reservation = GPU_RESERVATION_USD if mode == "run" else CPU_RESERVATION_USD
    guard = (
        compute_reservation_guard(snapshot["rates"])
        if mode == "run"
        else cpu_reservation_guard(snapshot["rates"], mode)
    )
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


def cpu_reservation_guard(rates, mode):
    """CPU 執行、核准傳送與公開匯出仍計入 US$10 的同一本帳本。"""
    keys = ("cpu_hour_cost", "mem_gib_hour_cost")
    if not all(key in rates for key in keys):
        raise ValueError("CPU 用量檢查需要已確認的每小時 API 單價")
    cpu, memory = (Decimal(rates[key]) for key in keys)
    if any(not value.is_finite() or value < 0 for value in (cpu, memory)):
        raise ValueError("CPU/記憶體單價必須為有限非負值")
    # 全部包含最長 timeout + 2 秒 idle；先驗證／最後帳本各一次。
    cost = Decimal(182) * (cpu / 4 + memory / 2) + Decimal(62) * (cpu / 4 + memory / 2)
    if mode == "run-cpu":
        cost += Decimal(602) * (2 * cpu + 8 * memory) + Decimal(602) * (cpu / 4 + memory)
    elif mode == "release":
        cost += Decimal(602) * (cpu / 4 + memory) + Decimal(602) * (cpu + 4 * memory)
    elif mode != "preflight":
        raise ValueError("未知 CPU 工作 mode")
    cost /= Decimal(3600)
    if cost > CPU_RESERVATION_USD:
        raise RuntimeError("CPU 工作保守估算超過 US$0.04 保留額；未啟動工作")
    return {"compute_upper_bound_usd": str(cost), "reserved_usd": str(CPU_RESERVATION_USD), "gpu_used": False}


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


def run_reserved(experiment_id, batch_id, run_id, revision, device, reservation_mode):
    from scripts.course_experiments.run import execute

    volume.reload()
    ledger = json.loads((VOLUME_ROOT / "budget.json").read_text())
    reservation = next(
        (
            entry
            for entry in ledger["reservations"]
            if entry["run_id"] == run_id
            and entry["mode"] == reservation_mode
            and entry["batch_id"] == batch_id
            and entry["experiment_id"] == experiment_id
            and entry["status"] == "reserved"
        ),
        None,
    )
    if reservation is None:
        raise RuntimeError("工作必須先完成相符的本任務預算保留")
    reservation["status"] = "running"
    write_json(VOLUME_ROOT / "budget.json", ledger)
    directory = VOLUME_ROOT / batch_id / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    for stale in ("result.json", "result-attestation.json", "failure.json", "approved-public-exports.json"):
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
            device=device,
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
        write_json(
            directory / "result-attestation.json",
            {
                "run_id": run_id,
                "revision": revision,
                "result_sha256": sha256(directory / "result.json"),
                "device": device,
                "proof": "authenticated Modal execution plus immutable private HF backup; SHA-256 is not a digital signature",
            },
        )
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
    return run_reserved(experiment_id, batch_id, run_id, revision, "cuda", "run")


@app.function(
    image=gpu_image,
    cpu=(2, 2),
    memory=(8192, 8192),
    volumes={"/course": volume},
    timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def train_cpu(experiment_id, batch_id, run_id, revision):
    if experiment_id != "simple_models":
        raise ValueError("run-cpu 目前只開放已規劃的 simple_models")
    return run_reserved(experiment_id, batch_id, run_id, revision, "cpu", "run-cpu")


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
            if key in ("private_only_artifacts", "private_only_data"):
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
            raise ValueError(f"禁止公開 private_only_artifacts/private_only_data：{relative}")
        if item.get("redistribution_approved") is not True or not item.get("license"):
            raise ValueError(f"缺少公開授權：{relative}")
        if relative.suffix in (".pt", ".pth", ".ckpt", ".safetensors") and item.get("kind") != "checkpoint":
            raise ValueError("權重檔必須經 checkpoint 推論匯出器，不能直接當 metadata 複製")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"審閱後檔案已變更：{relative}")
    return files


def inference_payload(saved, provenance):
    from scripts.course_release import inference_payload as export_payload

    return export_payload(saved, provenance)


@app.function(
    image=control_image,
    cpu=(0.25, 0.25),
    memory=(1024, 1024),
    secrets=[hf_secret],
    volumes={"/course": volume},
    timeout=600,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def stage_approval(checkpoint_repo, experiment_id, batch_id, approval_text, approval_hash, approval_git_revision):
    from huggingface_hub import HfApi, hf_hub_download

    from scripts.course_release import validate_approval

    if hashlib.sha256(approval_text.encode()).hexdigest() != approval_hash:
        raise ValueError("Actions 傳送的審閱清單 SHA-256 不相符")
    if not re.fullmatch(r"[0-9a-f]{40}", approval_git_revision):
        raise ValueError("需要已提交的審閱程式版本")
    approval = json.loads(approval_text)
    source = validate_approval(approval, experiment_id, batch_id, checkpoint_repo)
    volume.reload()
    stage = VOLUME_ROOT / batch_id / experiment_id / "release-source" / approval_hash
    api = HfApi(token=os.environ["HF_TOKEN"])
    with tempfile.TemporaryDirectory(prefix="approved-private-") as temporary:
        downloaded = hf_hub_download(
            repo_id=checkpoint_repo,
            token=os.environ["HF_TOKEN"],
            filename=f"{source['prefix']}/result.json",
            revision=source["revision"],
            local_dir=temporary,
        )
        result = json.loads(Path(downloaded).read_text())
        if (
            result.get("status") != "completed"
            or result.get("evidence_status") != "complete_run"
            or result.get("experiment_id") != experiment_id
            or result.get("revision") != approval["revision"]
        ):
            raise ValueError("固定 HF result 與已完成的審閱訓練不相符")
        artifacts = {item["path"]: item for item in result["artifacts"]}
        stage.mkdir(parents=True, exist_ok=True)
        for item in approval["files"]:
            if artifacts.get(item["path"], {}).get("sha256") != item["sha256"]:
                raise ValueError(f"核准 SHA 與私有訓練 result 不相符：{item['path']}")
            file = hf_hub_download(
                repo_id=checkpoint_repo,
                token=os.environ["HF_TOKEN"],
                filename=f"{source['prefix']}/{item['path']}",
                revision=source["revision"],
                local_dir=temporary,
            )
            if sha256(file) != item["sha256"]:
                raise ValueError(f"核准檔案與 HF 固定版本不相符：{item['path']}")
            target = stage / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
            base = item.get("base_checkpoint")
            if base:
                if api.repo_info(base["repo"], repo_type="model").private:
                    raise ValueError("公開 adapter 的 base_checkpoint 也必須公開")
                base_file = hf_hub_download(
                    repo_id=base["repo"],
                    token=False,
                    filename=base["filename"],
                    revision=base["revision"],
                    local_dir=temporary,
                )
                if sha256(base_file) != base["sha256"]:
                    raise ValueError("adapter 的公開基模 SHA-256 不相符")
                staged_base = stage / ".verified-bases" / f"{base['sha256']}.pt"
                staged_base.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base_file, staged_base)
        approved_files(stage, approval, result)
        write_json(stage / "result.json", result)
        (stage / "approved-public-exports.json").write_text(approval_text, encoding="utf-8")
        write_json(
            stage / "approval-attestation.json",
            {
                "approval_sha256": approval_hash,
                "approval_git_revision": approval_git_revision,
                "source_result_sha256": sha256(downloaded),
                "private_source": source,
            },
        )
    volume.commit()
    return {
        "approval_sha256": approval_hash,
        "approval_git_revision": approval_git_revision,
        "private_source": source,
        "files_verified": len(approval["files"]),
    }


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
def release(checkpoint_repo, release_repo, experiment_id, batch_id, run_id, approval_hash):
    from huggingface_hub import HfApi, hf_hub_download

    from scripts.course_release import build_export, validate_approval, write_model_card

    volume.reload()
    directory = VOLUME_ROOT / batch_id / experiment_id / "release-source" / approval_hash
    result = json.loads((directory / "result.json").read_text())
    if result.get("status") != "completed":
        raise ValueError("未完成的實驗不能發布")
    approval_path = directory / "approved-public-exports.json"
    if not approval_path.is_file():
        raise ValueError("尚未審閱：缺少 approved-public-exports.json；保留私有，不重訓")
    approval = json.loads(approval_path.read_text())
    if sha256(approval_path) != approval_hash:
        raise ValueError("Volume 的核准清單已被變更")
    validate_approval(approval, experiment_id, batch_id, checkpoint_repo)
    approved_files(directory, approval, result)
    attestation = json.loads((directory / "approval-attestation.json").read_text())
    provenance = {
        "revision": result["revision"],
        "experiment_id": experiment_id,
        "batch_id": batch_id,
        "approval_git_revision": attestation["approval_git_revision"],
        "approval_sha256": approval_hash,
        "scope": approval["model_card"]["scope"],
    }
    prefix = f"course/{batch_id}/{experiment_id}"
    api = HfApi(token=os.environ["HF_TOKEN"])
    with tempfile.TemporaryDirectory(prefix="course-public-") as temporary:
        export = Path(temporary)
        build_export(directory, export, approval, provenance)
        write_model_card(export / "README.md", approval, release_repo, prefix)
        weights_commit = api.upload_folder(
            repo_id=release_repo,
            folder_path=temporary,
            path_in_repo=prefix,
            commit_message=f"Publish reviewed inference files: {run_id}",
        )
        weight_files = [
            {
                "path": f"{prefix}/{path.relative_to(export).as_posix()}",
                "output": path.relative_to(export).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(export.rglob("*"))
            if path.is_file() and path.name != "README.md"
        ]
        write_json(
            export / "download-manifest.json",
            {
                "schema_version": 1,
                "id": experiment_id,
                "repo": release_repo,
                "revision": weights_commit.oid,
                "files": weight_files,
                "inference": approval.get("inference", {}),
            },
        )
        write_model_card(export / "README.md", approval, release_repo, prefix, weights_commit.oid)
        final_commit = api.upload_folder(
            repo_id=release_repo,
            folder_path=temporary,
            path_in_repo=prefix,
            allow_patterns=["README.md", "download-manifest.json"],
            commit_message=f"Pin model card and download hashes: {run_id}",
        )
        files = [
            {
                "path": f"{prefix}/{path.relative_to(export).as_posix()}",
                "output": path.relative_to(export).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(export.rglob("*"))
            if path.is_file()
        ]
        public_manifest = {
            "id": experiment_id,
            "repo": release_repo,
            "revision": final_commit.oid,
            "files": files,
            "inference": approval.get("inference", {}),
        }
        with tempfile.TemporaryDirectory(prefix="public-download-") as download:
            for item in files:
                file = hf_hub_download(
                    repo_id=release_repo,
                    token=False,
                    filename=item["path"],
                    revision=final_commit.oid,
                    local_dir=download,
                )
                if sha256(file) != item["sha256"]:
                    raise RuntimeError(f"學生匿名下載的公開檔案 SHA 不一致：{item['output']}")
        response = {
            "experiment_id": experiment_id,
            "repo": release_repo,
            "revision": final_commit.oid,
            "weights_revision": weights_commit.oid,
            "public_manifest": public_manifest,
            "anonymous_download_verified": True,
            "gpu_used": False,
        }
        if "repository_index" in approval:
            # 全文已由同一份 Git 核准清單綁定；權重及其匿名 roundtrip 成功後才更新根索引。
            index = approval["repository_index"]
            index_commit = api.upload_file(
                repo_id=release_repo,
                path_or_fileobj=index["content"].encode("utf-8"),
                path_in_repo="README.md",
                commit_message=f"Publish reviewed model index: {run_id}",
            )
            with tempfile.TemporaryDirectory(prefix="public-index-download-") as download:
                index_file = hf_hub_download(
                    repo_id=release_repo,
                    token=False,
                    filename="README.md",
                    revision=index_commit.oid,
                    local_dir=download,
                )
                if sha256(index_file) != index["sha256"]:
                    raise RuntimeError("學生匿名下載的根 README SHA 與核准索引不一致")
            response.update(
                root_index_revision=index_commit.oid,
                root_index_sha256=index["sha256"],
                root_index_anonymous_download_verified=True,
            )
        write_json(VOLUME_ROOT / batch_id / experiment_id / "public-release.json", response)
        volume.commit()
    return json.dumps(response, ensure_ascii=False, allow_nan=False)


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
    approval_file: str = "",
):
    for value in (experiment_id, batch_id, run_id):
        safe_name(value)
    if experiment_id == "preflight":
        mode = "preflight"
    if mode not in ("preflight", "run", "run-cpu", "release"):
        raise ValueError("mode 必須是 preflight/run/run-cpu/release")
    if mode == "run-cpu" and experiment_id != "simple_models":
        raise ValueError("run-cpu 只供 simple_models 使用")
    for repo in (checkpoint_repo, release_repo):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("HF repo 必須是 owner/name")
    approval_text, approval_hash = "", ""
    if mode == "release":
        from scripts.course_release import validate_approval

        relative = Path(approval_file or f"docs/course-experiments/releases/{experiment_id}.json")
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.as_posix().startswith("docs/course-experiments/releases/")
        ):
            raise ValueError("審閱清單必須在 repo 的 docs/course-experiments/releases/ 下")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("公開審閱需要完整的 checkout 程式 revision")
        committed = subprocess.run(
            ["git", "show", f"{revision}:{relative.as_posix()}"], cwd=ROOT, check=True, capture_output=True
        ).stdout
        if (ROOT / relative).read_bytes() != committed:
            raise ValueError("審閱清單必須與指定 git commit 完全相同，不能發布未提交的修改")
        approval_text = committed.decode("utf-8")
        validate_approval(json.loads(approval_text), experiment_id, batch_id, checkpoint_repo)
        approval_hash = hashlib.sha256(committed).hexdigest()
    output = ROOT / "outputs/modal-course"
    before = billing_snapshot()
    write_json(output / "billing-before.json", before)
    access = preflight.remote(checkpoint_repo, release_repo, run_id, batch_id, experiment_id, mode, revision, before)
    result, error = {"experiment_id": experiment_id, "mode": mode, "preflight": access}, None
    try:
        if mode in ("run", "run-cpu"):
            try:
                runner = train_cpu if mode == "run-cpu" else train
                runner.remote(experiment_id, batch_id, run_id, revision)
            finally:
                result = json.loads(backup.remote(checkpoint_repo, experiment_id, batch_id, run_id, revision))
                result["preflight"] = access
        elif mode == "release":
            result["approval"] = stage_approval.remote(
                checkpoint_repo, experiment_id, batch_id, approval_text, approval_hash, revision
            )
            result["release"] = json.loads(
                release.remote(checkpoint_repo, release_repo, experiment_id, batch_id, run_id, approval_hash)
            )
            write_json(output / "public-manifest.json", result["release"]["public_manifest"])
            write_json(output / "release.json", result["release"])
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
