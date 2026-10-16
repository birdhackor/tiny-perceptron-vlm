"""不依賴Torch的成對checkpoint封存；一個HF commit包含同一組best/last。"""

import hashlib
import json
import re
import shutil
from pathlib import Path

FILES = ("best.pt", "last.pt", "config.json", "tokenizer.json", "progress.json")
DATA_MANIFEST = "docs/lesson-products/data-v2-manifest.json"
TRANSFER_FILES = (
    *FILES,
    "model.pt",
    "training-receipt.json",
    "checkpoint-bundle.json",
    "checkpoint-backup.json",
    "reservation.json",
)
TRAIN_SOURCES = tuple(
    "tiny_perceptron/lesson_products/" + name + ".py" for name in ("data", "model", "training", "train")
)
RECIPE_KEYS = ("product", "mode", "batch_size", "seed", "learning_rate", "depth_weights")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle(directory, run_id, step, hashes, job):
    directory = Path(directory)
    value = {
        "version": 1,
        "run_id": run_id,
        "step": step,
        "data_manifest_sha256": hashes[DATA_MANIFEST],
        "source_hashes": {name: hashes[name] for name in TRAIN_SOURCES},
        "training_recipe": {key: job[key] for key in RECIPE_KEYS},
        "files": {
            name: {"sha256": sha(directory / name), "bytes": (directory / name).stat().st_size} for name in FILES
        },
        "scope": "Paired best/last with optimizer and RNG; hashes of one completed checkpoint boundary. A later failed run does not erase this boundary.",
    }
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path = directory / "checkpoint-bundle.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return value


def validate_resume(resume):
    if resume is None:
        return
    if not isinstance(resume, dict) or set(resume) != {"run_id", "source", "hf_revision", "bundle_sha256"}:
        raise ValueError("resume需明確指定原run、volume/hf、固定HF版本及bundle SHA")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", resume["run_id"]) or resume["source"] not in ("volume", "hf"):
        raise ValueError("resume來源不合法")
    if (resume["source"] == "hf" and not re.fullmatch(r"[a-f0-9]{40}", resume["hf_revision"] or "")) or (
        resume["source"] == "volume" and resume["hf_revision"] is not None
    ):
        raise ValueError("HF恢復需固定commit；Volume不冒稱HF版本")
    if not re.fullmatch(r"[a-f0-9]{64}", resume["bundle_sha256"]):
        raise ValueError("resume需要已核對的bundle SHA")


def prepare_resume(resume, destination, course_root, checkpoint_repo, hashes, job, ledger):
    """在CPU預檢控制容器完成讀取與SHA核對；GPU只讀已驗證的小檔。"""
    validate_resume(resume)
    if resume is None:
        return None
    original = next((e for e in ledger["reservations"] if e["run_id"] == resume["run_id"]), None)
    if (
        not original
        or original.get("experiment_id") != "selftrained_lesson_products"
        or original["status"] not in ("completed", "failed", "failed-or-cancelled")
    ):
        raise ValueError("只能恢復已確認結束的本成品attempt，不可與仍跑的工作並行")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    def source(name):
        if resume["source"] == "volume":
            return Path(course_root) / "lesson-products" / resume["run_id"] / name
        import os

        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                checkpoint_repo,
                f"lesson-products/{resume['run_id']}/{name}",
                revision=resume["hf_revision"],
                token=os.environ["HF_TOKEN"],
            )
        )

    raw = source("checkpoint-bundle.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != resume["bundle_sha256"]:
        raise ValueError("原checkpoint bundle不符已提交描述")
    value = json.loads(raw)
    if (
        value.get("version") != 1
        or value.get("run_id") != resume["run_id"]
        or set(value["files"]) != set(FILES)
        or value["source_hashes"] != {name: hashes[name] for name in TRAIN_SOURCES}
        or value["training_recipe"] != {key: job[key] for key in RECIPE_KEYS}
        or value.get("data_manifest_sha256") != hashes[DATA_MANIFEST]
        or original.get("manifest_sha256") != hashes[DATA_MANIFEST]
        or original.get("source_sha256", {}).get(DATA_MANIFEST) != hashes[DATA_MANIFEST]
        or any(original.get("source_sha256", {}).get(name) != value["source_hashes"][name] for name in TRAIN_SOURCES)
        or any(original.get("runtime_options", {}).get(key) != value["training_recipe"][key] for key in RECIPE_KEYS)
    ):
        raise ValueError("續訓來源的程式或配方不符")
    for name, expected in value["files"].items():
        path = source(name)
        if path.is_symlink() and resume["source"] == "volume":
            raise ValueError("Volume checkpoint不可是符號連結")
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("checkpoint bundle內容指紋不符：" + name)
        shutil.copyfile(path, destination / name)
    (destination / "checkpoint-bundle.json").write_bytes(raw)
    return {"source": resume, "step": value["step"], "files_verified": len(FILES)}


def transfer_payload(directory):
    directory = Path(directory)
    payload = {name: (directory / name).read_bytes() for name in TRANSFER_FILES if (directory / name).is_file()}
    if sum(map(len, payload.values())) > 20 * 1024 * 1024:
        raise ValueError("成品回傳超過預先界定20MiB")
    hashes = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} for name, raw in payload.items()}
    return payload, hashes


def save_transferred(envelope, destination):
    """RPC binary只存忽略目錄，receipt/帳本/日誌均不嵌入權重。"""
    expected = envelope["receipt"]["client_artifacts"]
    payload = envelope["artifacts"]
    if (
        set(payload) != set(expected)
        or not set(payload) <= set(TRANSFER_FILES)
        or sum(map(len, payload.values())) > 20 * 1024 * 1024
    ):
        raise ValueError("回傳artifact集合或大小不合法")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, raw in payload.items():
        path = destination / name
        if (
            path.is_symlink()
            or len(raw) != expected[name]["bytes"]
            or hashlib.sha256(raw).hexdigest() != expected[name]["sha256"]
        ):
            raise ValueError("回傳artifact指紋不符：" + name)
        path.write_bytes(raw)
    return expected
