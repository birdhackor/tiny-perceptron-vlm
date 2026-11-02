"""由 GitHub Actions 發布已驗收的固定 C-v3 包，不執行訓練。"""

import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "birdhackor/tiny-perceptron-vlm"
TAG = "c-v3-20261101"
RELEASE_ID = 430609439
NAME = "tiny-perceptron-c-v3-verified.zip"
SIZE = 6168350
DIGEST = "552d64b788291ce19157680f51dfc6bbc5394fb6bed7a22fcaa11f2ad3615d88"
DESCRIPTOR_DIGEST = "ac8210926a00700a3756aa73f5e877ca292ec7f3c6dfae9c096e45dbe99c3859"
PACKAGE = ROOT / "assets/releases/c-v3-20261101" / NAME
URL = f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{NAME}"


def verify_file(path, size=SIZE, digest=DIGEST):
    if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("固定模型包的大小或 SHA-256 不符；不呼叫發布 API")


def gh(arguments, payload=None):
    result = subprocess.run(
        ["gh", *arguments],
        input=None if payload is None else json.dumps(payload).encode(),
        capture_output=True,
        timeout=180,
        check=False,
    )
    if result.returncode:
        # 不把可能含重新導向網址的原始 stderr 寫進公開收據。
        raise RuntimeError("GitHub Release 操作失敗；請核對收據中的 stage")
    return result.stdout


def api(suffix, payload=None):
    args = ["api", f"repos/{REPOSITORY}/{suffix}"]
    if payload is not None:
        args += ["--method", "PATCH", "--input", "-"]
    return json.loads(gh(args, payload))


def checked_release(release, revision):
    if release.get("id") != RELEASE_ID or release.get("tag_name") != TAG:
        raise ValueError("只允許已指定的 C-v3 Release")
    if type(release.get("draft")) is not bool:
        raise ValueError("Release 狀態不明")
    if not release["draft"] and release.get("target_commitish") != revision:
        raise ValueError("已公開版本指向其他提交；不移動 tag 或覆寫既有版本")
    matches = [asset for asset in release.get("assets", []) if asset.get("name") == NAME]
    if len(matches) > 1:
        raise ValueError("同名模型附件不唯一")
    if matches and matches[0].get("size") != SIZE:
        raise ValueError("同名模型附件大小不同；不覆寫")
    return matches[0] if matches else None


def verify_implementation(root, descriptor_path):
    # 固定完整描述檔的原始 bytes，不能藉刪除項目跳過某個已驗收實作。
    raw = descriptor_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESCRIPTOR_DIGEST:
        raise ValueError("已驗收的完整17檔描述已改變")
    descriptor = json.loads(raw)
    for name, digest in descriptor["implementation"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("成品實作不等於已驗收的17個原始檔")
    return descriptor


def check_tag_response(status, reference, revision, draft):
    if status == 404 and draft is True:
        return False
    if status != 200:
        raise ValueError("tag查詢狀態不明或公開版本缺tag；停止發布")
    if (
        reference.get("ref") != f"refs/tags/{TAG}"
        or reference.get("object", {}).get("type") != "commit"
        or reference["object"]["sha"] != revision
    ):
        raise ValueError("同名tag不是本次真實成品提交；不移動tag、不發布")
    return True


def tag_preflight(revision, draft):
    # 從HTTP狀態行判404，不把權限／網路錯誤當成tag不存在。
    result = subprocess.run(
        ["gh", "api", "--include", f"repos/{REPOSITORY}/git/ref/tags/{TAG}"],
        capture_output=True,
        timeout=120,
        check=False,
    )
    header, separator, body = result.stdout.partition(b"\r\n\r\n")
    if not separator:
        header, separator, body = result.stdout.partition(b"\n\n")
    first = header.splitlines()[0] if header else b""
    match = re.fullmatch(rb"HTTP/\S+ (\d{3})(?: .*)?", first)
    if not separator or match is None:
        raise ValueError("tag查詢沒有可辨識HTTP狀態；停止發布")
    status = int(match[1])
    if result.returncode and status != 404:
        raise ValueError("tag查詢失敗；停止發布")
    reference = json.loads(body) if status == 200 else {}
    return check_tag_response(status, reference, revision, draft)


def publish():
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/revision/c-product-v3-20261101"
    ):
        raise ValueError("僅允許指定 repository／成品分支的 Actions runner")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or revision != os.environ.get("GITHUB_SHA"):
        raise ValueError("需要 runner 真正 checkout 的提交")
    verify_file(PACKAGE)
    descriptor = verify_implementation(ROOT, ROOT / "docs/decision-product/v3/public-release.json")
    if (
        descriptor["repository"] != REPOSITORY
        or descriptor["release_tag"] != TAG
        or descriptor["name"] != NAME
        or descriptor["url"] != URL
        or descriptor["bytes"] != SIZE
        or descriptor["sha256"] != DIGEST
    ):
        raise ValueError("教材固定下載描述與已驗收包不符")
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "repository": REPOSITORY,
        "workflow_revision": revision,
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
        "release_id": RELEASE_ID,
        "tag": TAG,
        "url": URL,
        "bytes": SIZE,
        "sha256": DIGEST,
        "training_started": False,
        "success": False,
        "stage": "inspect_existing_release",
    }
    output = ROOT / "outputs/c-v3-release-publication/result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        release = api(f"releases/{RELEASE_ID}")
        asset = checked_release(release, revision)
        result["stage"] = "preflight_actual_tag_before_mutation"
        tag_preflight(revision, release["draft"])
        if release["draft"]:
            # 先綁定真實成品提交，再上傳；草稿尚未建立公開 tag。
            result["stage"] = "bind_actual_product_commit"
            release = api(f"releases/{RELEASE_ID}", {"target_commitish": revision})
        if asset is None:
            result["stage"] = "upload_fixed_package"
            gh(["release", "upload", TAG, str(PACKAGE), "--repo", REPOSITORY])
            release = api(f"releases/{RELEASE_ID}")
            asset = checked_release(release, revision)
        if asset is None:
            raise ValueError("上傳後沒有唯一模型附件")
        result.update(asset_id=asset["id"], stage="verify_authenticated_asset")
        with tempfile.TemporaryDirectory() as directory:
            downloaded = Path(directory) / NAME
            downloaded.write_bytes(
                gh(
                    [
                        "api",
                        f"repos/{REPOSITORY}/releases/assets/{asset['id']}",
                        "-H",
                        "Accept: application/octet-stream",
                    ]
                )
            )
            verify_file(downloaded)
            if release["draft"]:
                result["stage"] = "recheck_actual_tag_before_publication"
                tag_preflight(revision, True)
                result["stage"] = "publish_verified_release"
                api(f"releases/{RELEASE_ID}", {"draft": False, "target_commitish": revision})
            result["stage"] = "verify_anonymous_download"
            # urllib 不會讀取 GH_TOKEN；此請求沒有 Authorization header。
            request = urllib.request.Request(URL, headers={"User-Agent": "tiny-perceptron-c-v3-publication"})
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read(SIZE + 1)
            downloaded.write_bytes(raw)
            verify_file(downloaded)
        tag = api(f"git/ref/tags/{TAG}")
        if tag["object"]["type"] != "commit" or tag["object"]["sha"] != revision:
            raise ValueError("公開 tag 不是已驗收實作所在的真實成品提交")
        result.update(success=True, public_transport_verified=True, tag_revision=revision, stage="complete")
    finally:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    publish()
