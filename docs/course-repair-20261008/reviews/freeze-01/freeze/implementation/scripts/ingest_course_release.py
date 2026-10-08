"""以固定 Git 核准清單驗證成功發布結果，保存學生下載清單及精簡發布證據。"""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from scripts.course_release import GENERATED_NAMES, validate_approval

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_REPO = "birdhackor/tiny-perceptron-checkpoints"
RELEASE_REPO = "birdhackor/tiny-perceptron-course-models"
INFERENCE_FIELDS = {
    "script",
    "checkpoint",
    "prompt",
    "tokens",
    "temperature",
    "chat",
    "tokenizer",
    "adapter",
    "image",
    "audio",
    "color",
    "shape",
    "frequency",
    "device",
    "resample_audio",
    "json",
    "cache",
}


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必須是 object")
    return value


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value) or ".." in value:
        raise ValueError("experiment/batch id 必須是安全的單一名稱")
    return value


def _hash(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value):
        raise ValueError(f"{label} 必須是固定的完整 {length}-character SHA")
    return value


def _path(value):
    if not isinstance(value, str) or not value or any(ord(character) < 32 for character in value):
        raise ValueError("公開檔案路徑必須是非空文字，不能含控制字元")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value or path.as_posix() != value:
        raise ValueError("公開檔案路徑必須是安全且正規化的 repo-relative path")
    return value


def _repo(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("HF repo 必須是 owner/repo")
    return value


def _inference(value, outputs):
    value = _object(value, "inference")
    if set(value) - INFERENCE_FIELDS:
        raise ValueError("inference 只能含公開 CLI 設定，不能含原始資料或未知欄位")
    for item in value.values():
        if item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("inference 參數必須是 scalar")
    if value:
        script = _path(value.get("script"))
        if not script.startswith("scripts/") or not script.endswith(".py"):
            raise ValueError("inference 需要 scripts/*.py")
        for key in ("checkpoint", "tokenizer", "adapter"):
            if key in value and value[key] not in outputs:
                raise ValueError("inference 的 checkpoint/companion 必須在公開檔案清單")
        if "checkpoint" not in value:
            raise ValueError("inference 必須指定公開 checkpoint")
    # 只保存經 Git 核准的公開操作設定；拒絕 NaN/Infinity。
    return json.loads(json.dumps(value, allow_nan=False))


def _files(value, prefix):
    if not isinstance(value, list) or not value:
        raise ValueError("published files 必須是非空清單")
    paths, outputs, clean = set(), set(), []
    for item in value:
        item = _object(item, "published file")
        path, output = _path(item.get("path")), _path(item.get("output"))
        if path != f"{prefix}/{output}" or path in paths or output in outputs:
            raise ValueError("published files 的 experiment prefix 或重複路徑無效")
        digest = _hash(item.get("sha256"), 64, "公開檔案 SHA-256")
        if type(item.get("bytes")) is not int or item["bytes"] < 1:
            raise ValueError("公開檔案 bytes 必須是正整數")
        paths.add(path)
        outputs.add(output)
        clean.append({"path": path, "output": output, "sha256": digest, "bytes": item["bytes"]})
    if not {"LICENSE", "README.md", "export-manifest.json"}.issubset(outputs):
        raise ValueError("published files 缺少 mandatory LICENSE/README/export-manifest")
    return sorted(clean, key=lambda item: item["output"])


def _git_approval(repo_root, revision, experiment_id, digest):
    path = f"docs/course-experiments/releases/{experiment_id}.json"
    try:
        completed = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("指定 approval Git revision 沒有此 experiment 的核准檔；不使用工作目錄草稿") from error
    if hashlib.sha256(completed.stdout).hexdigest() != digest:
        raise ValueError("approval Git bytes 的 SHA-256 與發布 receipt 漂移")
    try:
        approval = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("固定 Git 核准清單不是有效 UTF-8 JSON") from error
    return _object(approval, "Git approval"), path


def validated_release(result, repo_root, plan, checkpoint_repo=CHECKPOINT_REPO, release_repo=RELEASE_REPO):
    """只信固定 Git blob 的審閱內容；遠端匿名下載證據仍標明是 release receipt。"""
    result = _object(result, "Actions result")
    if result.get("mode") != "release" or result.get("status", "completed") != "completed":
        raise ValueError("只接受成功的 mode=release Actions result")
    if any(key in result for key in ("exception_type", "error", "billing_error")):
        raise ValueError("失敗的 Actions result 不能加入公開下載清單")
    if plan.get("schema_version") != 1 or not isinstance(plan.get("sequence"), list) or not plan["sequence"]:
        raise ValueError("需要 schema_version=1 且非空 sequence 的課程 plan")
    experiment_id = _name(result.get("experiment_id"))
    batch_id = _name(plan.get("batch_id"))
    order = [_name(item["id"]) for item in plan["sequence"]]
    if len(set(order)) != len(order) or experiment_id not in order:
        raise ValueError("experiment 不在唯一且固定順序的課程 plan")
    attestation = _object(result.get("approval"), "approval receipt")
    git_revision = _hash(attestation.get("approval_git_revision"), 40, "approval Git revision")
    approval_hash = _hash(attestation.get("approval_sha256"), 64, "approval SHA-256")
    approval, approval_path = _git_approval(repo_root, git_revision, experiment_id, approval_hash)
    try:
        source = validate_approval(approval, experiment_id, batch_id, _repo(checkpoint_repo))
    except (KeyError, TypeError) as error:
        raise ValueError("Git 核准清單的 schema 無效") from error
    if attestation.get("private_source") != source:
        raise ValueError("approval receipt private_source 與 Git 核准清單不一致")
    _path(source["prefix"])
    if type(attestation.get("files_verified")) is not int or attestation["files_verified"] != len(approval["files"]):
        raise ValueError("approval receipt 的核驗檔數與已審閱清單不一致")
    release = _object(result.get("release"), "release receipt")
    if release.get("status", "completed") != "completed" or any(key in release for key in ("error", "exception_type")):
        raise ValueError("失敗的 release receipt 不能加入公開下載清單")
    if (
        release.get("experiment_id") != experiment_id
        or release.get("anonymous_download_verified") is not True
        or release.get("gpu_used") is not False
    ):
        raise ValueError("release ID 不一致，或缺少遠端匿名下載驗證／CPU-only 證據")
    repo = _repo(release.get("repo"))
    if repo != _repo(release_repo):
        raise ValueError("release HF repo 與選定的公開 repo 不一致")
    revision = _hash(release.get("revision"), 40, "public HF revision")
    weights_revision = _hash(release.get("weights_revision"), 40, "weights HF revision")
    public = _object(release.get("public_manifest"), "public_manifest")
    if public.get("id") != experiment_id or public.get("repo") != repo or public.get("revision") != revision:
        raise ValueError("result/release/public_manifest 的 ID、repo 或固定 HF revision 不一致")
    files = _files(public.get("files"), f"course/{batch_id}/{experiment_id}")
    by_output = {item["output"]: item for item in files}
    approved_outputs = {_path(item.get("output", item["path"])) for item in approval["files"]}
    if not approved_outputs.issubset(by_output):
        raise ValueError("published files 缺少 root reviewed 的 approved artifact")
    if set(by_output) - approved_outputs - GENERATED_NAMES:
        raise ValueError("published files 包含未經 root 審閱的 artifact")
    if not any(item["kind"] == "checkpoint" for item in approval["files"]):
        raise ValueError("公開模型至少需要一份已審閱 checkpoint")
    for item in approval["files"]:
        output = item.get("output", item["path"])
        if item["kind"] != "checkpoint" and by_output[output]["sha256"] != item["sha256"]:
            raise ValueError("直接複製的 approved companion/data/license SHA 已漂移")
    inference = _inference(approval.get("inference", {}), by_output)
    if public.get("inference", {}) != inference:
        raise ValueError("public inference 與固定 Git 核准設定不一致")
    spec = {"id": experiment_id, "repo": repo, "revision": revision, "files": files, "inference": inference}
    proof = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "batch_id": batch_id,
        "mode": "release",
        "approval": {
            "approval_git_revision": git_revision,
            "approval_sha256": approval_hash,
            "approval_path": approval_path,
            "private_source": {key: source[key] for key in ("repo", "revision", "prefix")},
            "files_verified": len(approval["files"]),
        },
        "release": {
            "experiment_id": experiment_id,
            "repo": repo,
            "revision": revision,
            "weights_revision": weights_revision,
            "public_manifest": spec,
            "anonymous_download_verified": True,
            "anonymous_verification_scope": "Remote release worker verified pinned public file hashes; local inference is a separate student check.",
            "gpu_used": False,
        },
    }
    index_keys = {"root_index_revision", "root_index_sha256", "root_index_anonymous_download_verified"}
    if "repository_index" in approval or index_keys & release.keys():
        if "repository_index" not in approval or release.get("root_index_anonymous_download_verified") is not True:
            raise ValueError("根索引缺少 Git 核准內容或遠端匿名下載驗證")
        index_revision = _hash(release.get("root_index_revision"), 40, "root index HF revision")
        if release.get("root_index_sha256") != approval["repository_index"]["sha256"]:
            raise ValueError("root index SHA 與 Git 核准全文不一致")
        proof["release"].update(
            root_index_revision=index_revision,
            root_index_sha256=release["root_index_sha256"],
            root_index_anonymous_download_verified=True,
        )
    return spec, proof, order


def _write_json(path, value):
    if path.is_symlink():
        raise ValueError("不能覆寫 symlink 的發布清單或證據")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temporary.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        name = Path(temporary.name)
    name.replace(path)


def ingest(
    result_path,
    *,
    repo_root=ROOT,
    manifest_path=None,
    proof_dir=None,
    plan_path=None,
    replace=False,
    checkpoint_repo=CHECKPOINT_REPO,
    release_repo=RELEASE_REPO,
):
    repo_root = Path(repo_root)
    manifest_path = Path(manifest_path) if manifest_path else repo_root / "docs/course-experiments/public-models.json"
    proof_dir = Path(proof_dir) if proof_dir else repo_root / "docs/course-experiments/public-releases"
    plan_path = Path(plan_path) if plan_path else repo_root / "docs/course-experiments/plan.json"
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    plan = _object(json.loads(plan_path.read_text(encoding="utf-8")), "plan")
    spec, proof, order = validated_release(result, repo_root, plan, checkpoint_repo, release_repo)
    if manifest_path.is_symlink() or proof_dir.is_symlink():
        raise ValueError("發布清單與證據目錄不能是 symlink")
    manifest = _object(
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"schema_version": 1, "models": []},
        "既有 public-models.json",
    )
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), list):
        raise ValueError("既有 public-models.json schema 無效")
    models = manifest["models"]
    ids = [_name(_object(item, "既有公開模型").get("id")) for item in models]
    if len(ids) != len(set(ids)) or set(ids) - set(order):
        raise ValueError("既有公開模型 id 重複或不在課程 plan")
    previous = next((item for item in models if item["id"] == spec["id"]), None)
    if previous and previous != spec and not replace:
        raise ValueError("同 ID 的固定 HF SHA 或 manifest 已變更；需明示 --replace")
    proof_path = proof_dir / f"{spec['id']}.json"
    if proof_path.is_symlink():
        raise ValueError("durable release proof 不能是 symlink")
    previous_proof = json.loads(proof_path.read_text(encoding="utf-8")) if proof_path.is_file() else None
    if previous_proof is not None and previous_proof != proof and not replace:
        raise ValueError("同 ID 的 durable proof 已存在不同發布／核准 SHA；需明示 --replace")
    updated = {
        "schema_version": 1,
        "models": sorted(
            [item for item in models if item["id"] != spec["id"]] + [spec], key=lambda item: order.index(item["id"])
        ),
    }
    proof_written, manifest_written = previous_proof != proof, updated != manifest
    # 先檢查所有條件再寫；proof 優先，意外中斷可以安全地重跑補齊 manifest。
    if proof_written:
        _write_json(proof_path, proof)
    if manifest_written:
        _write_json(manifest_path, updated)
    return {
        "id": spec["id"],
        "repo": spec["repo"],
        "revision": spec["revision"],
        "approval_sha256": proof["approval"]["approval_sha256"],
        "manifest": str(manifest_path),
        "proof": str(proof_path),
        "manifest_written": manifest_written,
        "proof_written": proof_written,
        "status": "unchanged" if not proof_written and not manifest_written else "ingested",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--proof-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--checkpoint-repo", default=CHECKPOINT_REPO)
    parser.add_argument("--release-repo", default=RELEASE_REPO)
    parser.add_argument("--replace", action="store_true", help="明確替換同 ID 不同固定 SHA 的既有公開版本與證據")
    args = parser.parse_args()
    print(
        json.dumps(
            ingest(
                args.result,
                repo_root=args.repo_root,
                manifest_path=args.manifest,
                proof_dir=args.proof_dir,
                plan_path=args.plan,
                replace=args.replace,
                checkpoint_repo=args.checkpoint_repo,
                release_repo=args.release_repo,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
