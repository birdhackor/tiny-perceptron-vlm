"""只把已核對的LFS包上傳到既有draft，並下載核對；不啟動Modal。"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_local(spec):
    name = Path(spec["asset_path"])
    if (
        spec["version"] != 1
        or spec["repo"] != "birdhackor/tiny-perceptron-vlm"
        or not re.fullmatch(r"[a-f0-9]{40}", spec["release_commit"])
        or name.parent.as_posix() != "assets/lesson-products"
        or name.name != spec["asset_name"]
        or name.suffixes != [".tar", ".gz"]
        or type(spec["release_id"]) is not int
        or spec["release_id"] <= 0
        or type(spec["bytes"]) is not int
        or not 0 < spec["bytes"] <= 64 * 1024 * 1024
        or not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"])
    ):
        raise ValueError("需要固定repo/draft/commit與有界LFS包")
    pointer = subprocess.check_output(["git", "show", "HEAD:" + name.as_posix()], cwd=ROOT).decode()
    expected = f"version https://git-lfs.github.com/spec/v1\noid sha256:{spec['sha256']}\nsize {spec['bytes']}\n"
    path = ROOT / name
    if (
        pointer != expected
        or path.is_symlink()
        or path.stat().st_size != spec["bytes"]
        or digest(path) != spec["sha256"]
    ):
        raise ValueError("Git必須只保存正確LFS pointer，工作檔大小／SHA也必須吻合")
    return path


def parent(spec):
    value = json.loads(subprocess.check_output(["gh", "api", f"repos/{spec['repo']}/releases/{spec['release_id']}"]))
    if value.get("draft") is not True or value.get("target_commitish") != spec["release_commit"]:
        raise ValueError("不是原本固定commit的draft")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="只核LFS與draft，不上傳")
    options = parser.parse_args()
    spec = json.loads(options.descriptor.read_bytes())
    if os.environ.get("GITHUB_REPOSITORY") != spec["repo"]:
        raise ValueError("只允許指定來源repo")
    path = verify_local(spec)
    before = parent(spec)
    if before["assets"]:
        raise ValueError("draft已含asset，拒絕重複上傳或覆寫")
    if options.dry_run:
        print(json.dumps({"status": "local-and-parent-verified", "sha256": spec["sha256"], "uploaded": False}))
        return
    subprocess.run(["gh", "release", "upload", before["tag_name"], str(path), "--repo", spec["repo"]], check=True)
    after = parent(spec)
    if len(after["assets"]) != 1:
        raise ValueError("draft asset inventory不符")
    asset = after["assets"][0]
    if asset["name"] != spec["asset_name"] or asset["size"] != spec["bytes"]:
        raise ValueError("實際asset名稱或大小不符")
    destination = ROOT / "outputs/lesson-products-transport"
    destination.mkdir(parents=True, exist_ok=False)
    download = destination / "download.tar.gz"
    with download.open("xb") as stream:
        subprocess.run(
            [
                "gh",
                "api",
                f"repos/{spec['repo']}/releases/assets/{asset['id']}",
                "-H",
                "Accept: application/octet-stream",
            ],
            stdout=stream,
            check=True,
        )
    if download.stat().st_size != spec["bytes"] or digest(download) != spec["sha256"]:
        raise ValueError("實際GitHub下載的asset與LFS原包不符")
    receipt = {
        "status": "completed",
        "code_revision": os.environ["GITHUB_SHA"],
        "release_id": spec["release_id"],
        "release_commit": spec["release_commit"],
        "asset_id": asset["id"],
        "asset_name": asset["name"],
        "asset_bytes": asset["size"],
        "transport_sha256": spec["sha256"],
        "download_sha256": digest(download),
        "modal_started": False,
        "source": "SHA-verified LFS source; genuine GitHub upload and pinned asset download roundtrip",
    }
    (destination / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
