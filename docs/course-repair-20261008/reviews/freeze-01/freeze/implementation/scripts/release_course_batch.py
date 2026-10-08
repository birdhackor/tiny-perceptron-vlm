"""先核對整批已提交核准清單，再依序發布；不訓練、不並行、不自動重試。"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.course_release import validate_approval  # noqa: E402
from scripts.ingest_course_release import validated_release  # noqa: E402

GENERATED_FILES = ("billing-before.json", "billing-after.json", "result.json", "release.json", "public-manifest.json")
ITEM_TIMEOUT_SECONDS = 1800


def parse_experiments(value):
    """只接受 JSON 字串清單或逗號清單；空值、重複與特殊路徑都拒絕。"""
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError("experiment 清單必須是至多 4096 字元的文字")
    value = value.strip()
    items = json.loads(value) if value.startswith("[") else [part.strip() for part in value.split(",")]
    if not isinstance(items, list) or not 1 <= len(items) <= 30:
        raise ValueError("每批需指定 1 至 30 個 experiment ID")
    if any(not isinstance(item, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", item) for item in items):
        raise ValueError("experiment ID 只能含小寫字母、數字與底線，並以字母開頭")
    if len(set(items)) != len(items):
        raise ValueError("同一批不能重複發布相同 experiment")
    return items


def _now():
    return datetime.now(UTC).isoformat()


def _write_json(path, value):
    if path.is_symlink():
        raise ValueError("不能覆寫 symlink 證據")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _directory(path, root):
    """不沿著 outputs 中的 symlink 寫入或清除其它位置。"""
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("輸出目錄不能使用 symlink")
    path.mkdir(parents=True, exist_ok=True)


def _committed_bytes(root, revision, relative):
    committed = subprocess.run(
        ["git", "show", f"{revision}:{relative}"], cwd=root, check=True, capture_output=True
    ).stdout
    working = root / relative
    if working.is_symlink() or working.read_bytes() != committed:
        raise ValueError("工作檔必須與指定 Git commit 的 bytes 完全相同")
    return committed


def validate_batch(experiments, *, root, revision, batch_id, checkpoint_repo, release_repo, run_prefix):
    """所有本機核准檢查都在第一個 Modal 呼叫之前完成。"""
    if parse_experiments(json.dumps(experiments)) != experiments:
        raise ValueError("無效 experiment 清單")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("需要完整的 GITHUB_SHA")
    if os.environ.get("GITHUB_SHA") and os.environ["GITHUB_SHA"] != revision:
        raise ValueError("revision 必須等於本次 Actions 的 GITHUB_SHA")
    if not isinstance(batch_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", batch_id):
        raise ValueError("batch ID 必須是安全的單一名稱")
    if (
        ".." in batch_id
        or not isinstance(run_prefix, str)
        or not re.fullmatch(r"gha-[1-9][0-9]*-[1-9][0-9]*", run_prefix)
    ):
        raise ValueError("需要安全的 batch ID 與 gha-RUN_ID-RUN_ATTEMPT 前綴")
    if os.environ.get("GITHUB_RUN_ID") and os.environ.get("GITHUB_RUN_ATTEMPT"):
        expected = f"gha-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
        if run_prefix != expected:
            raise ValueError("run ID 前綴必須屬於本次 Actions run／attempt")
    for repo in (checkpoint_repo, release_repo):
        if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("HF repo 必須是 owner/name")
    if checkpoint_repo == release_repo:
        raise ValueError("私有 checkpoint repo 與公開 release repo 不能相同")
    plan = json.loads(_committed_bytes(root, revision, "docs/course-experiments/plan.json"))
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != 1
        or plan.get("batch_id") != batch_id
        or not isinstance(plan.get("sequence"), list)
        or not plan["sequence"]
    ):
        raise ValueError("batch 與已提交的課程 plan 不相符")
    planned = [item["id"] for item in plan["sequence"]]
    if len(set(planned)) != len(planned) or any(item not in planned or item == "preflight" for item in experiments):
        raise ValueError("只能發布已提交課程 plan 中的 experiment")
    approved = []
    for experiment in experiments:
        run_id = f"{run_prefix}-{experiment}"
        if len(run_id) > 100:
            raise ValueError("run ID 超過既有 Modal entrypoint 的長度限制")
        relative = f"docs/course-experiments/releases/{experiment}.json"
        content = _committed_bytes(root, revision, relative)
        # 與既有 Modal entrypoint 相同：UTF-8 字串解析不接受帶 BOM 的核准檔。
        approval = json.loads(content.decode("utf-8"))
        if not isinstance(approval, dict):
            raise ValueError("核准清單必須是 JSON object")
        validate_approval(approval, experiment, batch_id, checkpoint_repo)
        if not any(item["kind"] == "checkpoint" for item in approval["files"]):
            raise ValueError("每個學生模型發布項目至少需有一份已核准 checkpoint")
        approved.append(
            {
                "experiment_id": experiment,
                "run_id": run_id,
                "approval_file": relative,
                "approval_sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return approved, plan


def _archive_generated(source, destination):
    """只複製本次 release 的固定輸出；確認副本後才清除這五種檔案。"""
    evidence = []
    for name in GENERATED_FILES:
        path = source / name
        if path.is_symlink():
            raise ValueError("Modal 證據不能使用 symlink")
        if not path.exists():
            continue
        if not path.is_file():
            raise ValueError("Modal 證據必須是一般檔案")
        target = destination / name
        shutil.copyfile(path, target)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("證據副本 SHA 不一致；保留原檔並停止")
        evidence.append({"path": name, "bytes": target.stat().st_size, "sha256": digest})
    # 整批複製完成後才清除；其它檔案（例如 assets.txt）不動。
    for item in evidence:
        path = source / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("清理前 Modal 證據已改變；停止")
        path.unlink()
    return evidence


def _verify_completed(directory, item, *, root, revision, plan, checkpoint_repo, release_repo):
    if any(not (directory / name).is_file() for name in GENERATED_FILES):
        raise ValueError("成功發布必須留下 result、release、public manifest 與前後 billing 證據")
    result = json.loads((directory / "result.json").read_text())
    if result.get("experiment_id") != item["experiment_id"]:
        raise ValueError("發布結果不能屬於另一個 experiment")
    attestation = result.get("approval", {})
    if (
        attestation.get("approval_git_revision") != revision
        or attestation.get("approval_sha256") != item["approval_sha256"]
    ):
        raise ValueError("發布核准 receipt 必須綁定本批 Git SHA 與 approval bytes")
    entry = result.get("billing", {}).get("entry", {})
    expected = {"run_id": item["run_id"], "experiment_id": item["experiment_id"], "batch_id": plan["batch_id"]}
    if (
        any(entry.get(key) != value for key, value in expected.items())
        or entry.get("status") != "completed"
        or entry.get("mode") != "release"
        or entry.get("compute_guard", {}).get("gpu_used") is not False
        or Decimal(entry.get("reserved_usd", "-1")) != Decimal("0.04")
    ):
        raise ValueError("本項 CPU-only release 的帳本必須正確完成；不沿用其它 run")
    spec, proof, _order = validated_release(result, root, plan, checkpoint_repo, release_repo)
    if (
        json.loads((directory / "public-manifest.json").read_text()) != result["release"]["public_manifest"]
        or json.loads((directory / "release.json").read_text()) != result["release"]
    ):
        raise ValueError("分開保存的 public manifest／release receipt 與 result 不一致")
    _write_json(directory / "validated-release.json", proof)
    return spec


def run_batch(experiments, *, root=ROOT, revision, batch_id, checkpoint_repo, release_repo, run_prefix):
    root = Path(root).resolve()
    approved, plan = validate_batch(
        experiments,
        root=root,
        revision=revision,
        batch_id=batch_id,
        checkpoint_repo=checkpoint_repo,
        release_repo=release_repo,
        run_prefix=run_prefix,
    )
    output, modal_output = root / "outputs/course-release-batch", root / "outputs/modal-course"
    _directory(output, root)
    _directory(modal_output, root)
    if any(path.name != "validation.json" for path in output.iterdir()):
        raise ValueError("batch 輸出已有舊證據；請使用新的 Actions 執行，不能覆寫")
    validation = output / "validation.json"
    if validation.is_symlink() or (validation.exists() and not validation.is_file()):
        raise ValueError("validation 證據必須是一般檔案")
    if any((modal_output / name).exists() or (modal_output / name).is_symlink() for name in GENERATED_FILES):
        raise ValueError("modal-course 有既存 release 證據；不清除或混用")
    summary = {
        "schema_version": 1,
        "mode": "release",
        "revision": revision,
        "batch_id": batch_id,
        "run_prefix": run_prefix,
        "started_at": _now(),
        "status": "running",
        "requested": experiments,
        "items": [],
        "retries": 0,
        "gpu_requested": False,
        "not_started": list(experiments),
        "billing_scope": "Account-wide observations; not attributable to this project. Existing shared budget applies.",
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    for item in approved:
        directory = output / item["experiment_id"]
        directory.mkdir()
        command = [
            "modal",
            "run",
            "scripts/modal_course.py",
            "--mode",
            "release",
            "--experiment-id",
            item["experiment_id"],
            "--batch-id",
            batch_id,
            "--approval-file",
            item["approval_file"],
            "--checkpoint-repo",
            checkpoint_repo,
            "--release-repo",
            release_repo,
            "--run-id",
            item["run_id"],
            "--revision",
            revision,
        ]
        record = {**item, "status": "running", "started_at": _now(), "command": command, "evidence": []}
        summary["items"].append(record)
        summary["not_started"].remove(item["experiment_id"])
        _write_json(directory / "invocation.json", record)
        _write_json(summary_path, summary)
        failure = None
        try:
            with (
                (directory / "modal-stdout.txt").open("w") as stdout,
                (directory / "modal-stderr.txt").open("w") as stderr,
            ):
                completed = subprocess.run(
                    command, cwd=root, check=False, stdout=stdout, stderr=stderr, timeout=ITEM_TIMEOUT_SECONDS
                )
            record["returncode"] = completed.returncode
            if completed.returncode:
                raise RuntimeError("單項 Modal release 失敗；保留證據，不重試或發布下一項")
        except Exception as error:
            failure = error
        try:
            record["evidence"] = _archive_generated(modal_output, directory)
            if failure is None:
                public = _verify_completed(
                    directory,
                    item,
                    root=root,
                    revision=revision,
                    plan=plan,
                    checkpoint_repo=checkpoint_repo,
                    release_repo=release_repo,
                )
                record["public_revision"] = public["revision"]
        except Exception as error:
            failure = failure or error
        record.update(status="failed" if failure else "completed", finished_at=_now())
        if failure:
            record.update(error_type=type(failure).__name__, message=str(failure))
            summary.update(status="failed", finished_at=_now())
        _write_json(directory / "invocation.json", record)
        _write_json(summary_path, summary)
        if failure:
            raise RuntimeError("serial release 已停止；逐項證據與未執行項目見 batch summary") from failure
        print(f"已發布 {item['experiment_id']}：CPU-only、固定 HF revision 與匿名下載已核對。", flush=True)
    summary.update(status="completed", finished_at=_now())
    _write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", required=True, help='JSON array 或逗號清單，例如 ["audio","vqa"]')
    parser.add_argument("--batch-id", default=os.environ.get("BATCH_ID", "course-v1"))
    parser.add_argument("--revision", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--checkpoint-repo", default=os.environ.get("HF_CHECKPOINT_REPO"))
    parser.add_argument("--release-repo", default=os.environ.get("HF_RELEASE_REPO"))
    parser.add_argument(
        "--run-id-prefix",
        default=f"gha-{os.environ.get('GITHUB_RUN_ID', '')}-{os.environ.get('GITHUB_RUN_ATTEMPT', '')}",
    )
    parser.add_argument("--validate-only", action="store_true", help="只檢查整批 committed approval，不呼叫 Modal")
    args = parser.parse_args()
    settings = {key: getattr(args, key) for key in ("revision", "batch_id", "checkpoint_repo", "release_repo")}
    settings["run_prefix"] = args.run_id_prefix
    try:
        experiments = parse_experiments(args.experiments)
        if args.validate_only:
            approved, _plan = validate_batch(experiments, root=ROOT, **settings)
            print(
                json.dumps(
                    {"status": "validated", "revision": args.revision, "items": approved}, ensure_ascii=False, indent=2
                )
            )
        else:
            result = run_batch(experiments, **settings)
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "completed": len(result["items"]),
                        "summary": "outputs/course-release-batch/summary.json",
                    }
                )
            )
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(error).__name__, "message": str(error)}, ensure_ascii=False
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    # Windows 的隔離模式忽略 PYTHONUTF8；重導至 pipe 時中文 help／JSON 仍需 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
