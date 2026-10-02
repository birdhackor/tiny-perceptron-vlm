"""GitHub Actions → Modal L4 → HF checkpoint → 下載續訓的有界測試。"""

import json
import os
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("tiny-perceptron-gpu-smoke")
hf_secret = modal.Secret.from_name("tiny-perceptron-hf", required_keys=["HF_TOKEN"])
volume = modal.Volume.from_name("tiny-perceptron-checkpoints", create_if_missing=True)
cpu_image = modal.Image.debian_slim(python_version="3.13").pip_install("huggingface-hub==1.33.0")
# Modal SDK 是執行工具；模型依賴仍使用原有 uv.lock，CUDA 12.6 支援 L4。
gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(str(ROOT), extras=["cu126"], uv_version="0.12.22", extra_options="--no-dev")
    .env({"PYTHONPATH": "/app", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .workdir("/app")
    .add_local_dir(ROOT / "tiny_perceptron", "/app/tiny_perceptron", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "scripts", "/app/scripts", ignore=["**/__pycache__/**"])
)


@app.function(image=cpu_image, secrets=[hf_secret], timeout=180, retries=0, max_containers=1)
def preflight(checkpoint_repo, release_repo, run_id, revision):
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    private = api.repo_info(checkpoint_repo, repo_type="model").private
    if not private:
        raise RuntimeError("中途 checkpoint repo 應為 Private；先確認 repo 設定")
    api.repo_info(release_repo, repo_type="model")
    # 實際寫入一個小檔案，先驗證 HF 權限，再使用 GPU。
    api.upload_file(
        repo_id=checkpoint_repo,
        path_or_fileobj=json.dumps({"run_id": run_id, "revision": revision}).encode(),
        path_in_repo=f"smoke-tests/{run_id}/preflight.json",
        commit_message=f"Check Modal upload access: {run_id}",
    )
    return {"checkpoint_repo_private": private, "hf_read_and_write": True}


@app.function(
    image=gpu_image,
    gpu="L4",
    cpu=2,
    memory=4096,
    secrets=[hf_secret],
    volumes={"/checkpoints": volume},
    timeout=600,
    retries=0,
    max_containers=1,
)
def train(checkpoint_repo, run_id, revision):
    from huggingface_hub import HfApi, hf_hub_download

    from scripts.gpu_smoke_check import resume_and_compare, sha256, train_and_measure, write_json

    directory = Path("/checkpoints") / run_id
    result = train_and_measure(directory, revision=revision)
    volume.commit()
    api = HfApi(token=os.environ["HF_TOKEN"])
    prefix = f"smoke-tests/{run_id}/training"
    commit = api.upload_folder(
        repo_id=checkpoint_repo,
        folder_path=str(directory),
        path_in_repo=prefix,
        allow_patterns=[
            "checkpoint-40.pt",
            "config.json",
            "tokenizer.json",
            "data-manifest.json",
            "training-check.json",
        ],
        commit_message=f"Save GPU training step 40: {run_id}",
    )
    downloaded = hf_hub_download(
        repo_id=checkpoint_repo,
        filename=f"{prefix}/checkpoint-40.pt",
        revision=commit.oid,
        token=os.environ["HF_TOKEN"],
        local_dir=f"/tmp/hf-roundtrip-{run_id}",
        force_download=True,
    )
    if sha256(downloaded) != result["checkpoint_sha256"]:
        raise RuntimeError("HF 下載檔案的 SHA-256 與原始 checkpoint 不一致")
    result["hf"] = {
        "repo": checkpoint_repo,
        "revision": commit.oid,
        "checkpoint_path": f"{prefix}/checkpoint-40.pt",
        "sha256_verified": True,
    }
    result["resume"] = resume_and_compare(directory, downloaded)
    write_json(directory / "result.json", result)
    final_commit = api.upload_folder(
        repo_id=checkpoint_repo,
        folder_path=str(directory),
        path_in_repo=prefix,
        allow_patterns=["resumed-80.pt", "result.json"],
        commit_message=f"Save resumed GPU training step 80: {run_id}",
    )
    result["hf"]["final_revision"] = final_commit.oid
    write_json(directory / "result.json", result)
    volume.commit()
    return result


@app.function(image=cpu_image, volumes={"/checkpoints": volume}, timeout=60, retries=0, max_containers=1)
def verify_volume(run_id, expected_sha256):
    import hashlib

    volume.reload()
    checkpoint = Path("/checkpoints") / run_id / "checkpoint-40.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError("另一個 container 無法讀到正確的持久 checkpoint")
    return True


@app.local_entrypoint()
def main(checkpoint_repo: str, release_repo: str, run_id: str, revision: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id 只能包含英數字、底線、連字號")
    for repo in (checkpoint_repo, release_repo):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("HF repo 必須是 owner/name")
    access = preflight.remote(checkpoint_repo, release_repo, run_id, revision)
    result = train.remote(checkpoint_repo, run_id, revision)
    result["preflight"] = access
    result["volume_persisted"] = verify_volume.remote(run_id, result["checkpoint_sha256"])
    output = ROOT / "outputs/modal-smoke"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
