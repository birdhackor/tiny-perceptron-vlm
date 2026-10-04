"""匿名取得固定版本的自然照片、中文讀字與語音資料，逐檔核對後解包。"""

import argparse
import hashlib
import json
import re
import tarfile
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/natural-assistant/manifest.json"
DEFAULT_REVISION = "21a24124353487e08881302bbd71614766a489ac"
DEFAULT_MANIFEST_SHA256 = "7604526c67da31a41940f16f9c87027c73cf2782c63522dbde8c7efbf015db91"
REPOSITORY = "birdhackor/tiny-perceptron-vlm"
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_FILES = 10000
ARCHIVES = {
    "assets/training/natural-vision-v3.tar.gz": "vision",
    "assets/training/natural-ocr-v3.tar.gz": "ocr",
    "assets/training/natural-speech-v3.tar.gz": "speech",
}
SPLITS = ("train", "validation", "test")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_pin(value, length, label):
    if not isinstance(value, str) or not re.fullmatch(rf"[a-f0-9]{{{length}}}", value):
        raise ValueError(f"{label} 必須是固定的 {length} 位十六進位指紋。")


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("資料路徑必須是非空白的相對路徑。")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.as_posix() != value
        or any(part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in path.parts)
    ):
        raise ValueError("資料路徑不能跳出目標資料夾。")
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("資料清單的 JSON 欄位不能重複。")
        result[key] = value
    return result


def declarations(files):
    if not isinstance(files, list) or not 0 < len(files) <= MAX_FILES:
        raise ValueError("資料包需要有界、逐檔列出的內容清單。")
    result = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("每個資料檔案都需要路徑、大小與 SHA-256。")
        path = safe_relative(item.get("path")).as_posix()
        if path in result:
            raise ValueError("資料清單有重複的路徑。")
        if type(item.get("bytes")) is not int or not 0 <= item["bytes"] < MAX_TOTAL_BYTES:
            raise ValueError("每個資料檔案都需要有界的非負整數大小。")
        digest_pin(item.get("sha256"), 64, "資料 SHA-256")
        result[path] = {"path": path, "bytes": item["bytes"], "sha256": item["sha256"]}
    if sum(item["bytes"] for item in result.values()) >= MAX_TOTAL_BYTES:
        raise ValueError("解包資料必須小於 512 MiB。")
    return result


def validate_manifest(manifest):
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("自然資料清單需要 schema_version=1。")
    if not isinstance(manifest.get("dataset_version"), str) or not manifest["dataset_version"]:
        raise ValueError("資料清單必須標明版本。")
    files = declarations(manifest.get("files"))
    archives = manifest.get("archives")
    if not isinstance(archives, list) or len(archives) != 3:
        raise ValueError("此入口只處理明確列出的三個自然資料包。")
    seen, combined, total = set(), {}, 0
    for archive in archives:
        if (
            not isinstance(archive, dict)
            or not isinstance(archive.get("path"), str)
            or archive["path"] not in ARCHIVES
            or archive["path"] in seen
        ):
            raise ValueError("資料包路徑未知或重複。")
        seen.add(archive["path"])
        if type(archive.get("bytes")) is not int or not 0 < archive["bytes"] < MAX_TOTAL_BYTES:
            raise ValueError("壓縮包必須有界並記錄精確大小。")
        total += archive["bytes"]
        digest_pin(archive.get("sha256"), 64, "壓縮包 SHA-256")
        group = declarations(archive.get("files"))
        if any(safe_relative(path).parts[0] != ARCHIVES[archive["path"]] or path in combined for path in group):
            raise ValueError("各包逐檔清單必須位於自己的目錄，且不能重複。")
        combined.update(group)
    if total >= MAX_TOTAL_BYTES or combined != files:
        raise ValueError("壓縮包總量過大，或逐包清單與完整逐檔清單不同。")
    identifiers = set()
    for key in ("rows", "audio_rows"):
        rows = manifest.get(key)
        if not isinstance(rows, list) or len(rows) > MAX_FILES:
            raise ValueError("資料需要明確列出文字／圖片紀錄與語音紀錄。")
        for row in rows:
            if not isinstance(row, dict) or row.get("split") not in SPLITS:
                raise ValueError("每筆資料必須宣告 train、validation 或 test。")
            if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in identifiers:
                raise ValueError("每筆資料需要不重複的 ID。")
            identifiers.add(row["id"])
            assets = [row[field] for field in ("image", "audio") if row.get(field)]
            history = row.get("history", [])
            if not isinstance(history, list):
                raise ValueError("先前回合需要以清單保存。")
            for turn in history:
                if not isinstance(turn, dict):
                    raise ValueError("先前回合需要明確的內容欄位。")
                if isinstance(turn.get("content"), list):
                    for part in turn["content"]:
                        if not isinstance(part, dict):
                            raise ValueError("回合中的圖片與文字需要明確的內容欄位。")
                        if part.get("type") == "image":
                            assets.append(part.get("image"))
            if any(safe_relative(path).as_posix() not in files for path in assets):
                raise ValueError("資料紀錄引用了逐檔清單以外的檔案。")
    return manifest


def load_manifest(path, expected_sha256):
    digest_pin(expected_sha256, 64, "資料清單 SHA-256")
    path = Path(path)
    if not 0 < path.stat().st_size <= MAX_MANIFEST_BYTES or sha256(path) != expected_sha256:
        raise ValueError("本機資料清單與固定指紋不同；請勿自行改用最新版。")
    return validate_manifest(json.loads(path.read_bytes(), object_pairs_hook=unique_object))


def summary(manifest, revision, manifest_sha256):
    digest_pin(revision, 40, "Git 版本")
    rows = Counter(row["split"] for row in manifest["rows"])
    audio = Counter(row["split"] for row in manifest["audio_rows"])
    return {
        "dataset_version": manifest["dataset_version"],
        "git_revision": revision,
        "manifest_sha256": manifest_sha256,
        "rows": {split: rows[split] for split in SPLITS},
        "audio_rows": {split: audio[split] for split in SPLITS},
        "files": len(manifest["files"]),
        "download_bytes": sum(archive["bytes"] for archive in manifest["archives"]),
        "unpacked_bytes": sum(item["bytes"] for item in manifest["files"]),
        "archives": [
            {"id": ARCHIVES[archive["path"]], "bytes": archive["bytes"], "files": len(archive["files"])}
            for archive in manifest["archives"]
        ],
    }


def download_checked(url, destination, size, expected_sha256, *, opener=None):
    """Request no credentials, keep TLS checks, and bound every response byte."""
    if urlsplit(url).scheme != "https":
        raise ValueError("公開資料必須以 HTTPS 傳送。")
    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(
        url, headers={"User-Agent": "tiny-perceptron-natural-data/3", "Accept-Encoding": "identity"}
    )
    digest, count = hashlib.sha256(), 0
    with opener(request, timeout=30) as response:
        if urlsplit(response.geturl()).scheme != "https":
            raise ValueError("公開資料不能重新導向不安全的 HTTP 位址。")
        length = response.headers.get("Content-Length")
        if length is not None and (not length.isdecimal() or int(length) != size):
            raise ValueError("公開檔案回報的大小與固定清單不同。")
        if response.headers.get("Content-Encoding", "identity").lower() not in {"identity", ""}:
            raise ValueError("公開檔案使用了未宣告的額外傳輸壓縮。")
        with Path(destination).open("xb") as stream:
            while chunk := response.read(min(1024 * 1024, size - count + 1)):
                count += len(chunk)
                if count > size:
                    raise ValueError("公開檔案超過固定清單的大小。")
                digest.update(chunk)
                stream.write(chunk)
    if count != size or digest.hexdigest() != expected_sha256:
        raise ValueError("公開檔案的大小／SHA-256 不符。")
    return {"url": url, "bytes": count, "sha256": digest.hexdigest(), "authentication": "none"}


def extract_checked(archive_path, specification, directory):
    archive_path, directory = Path(archive_path), Path(directory)
    if archive_path.stat().st_size != specification["bytes"] or sha256(archive_path) != specification["sha256"]:
        raise ValueError("壓縮包與固定指紋不同，或目前只有 Git LFS pointer。")
    expected = declarations(specification["files"])
    seen = set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            path = safe_relative(member.name)
            if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE} or member.name in seen:
                raise ValueError("資料包不能含連結、裝置、目錄或重複檔案。")
            item = expected.get(member.name)
            if item is None or member.size != item["bytes"]:
                raise ValueError("壓縮包成員不在清單中，或大小不同。")
            target = directory.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, count = hashlib.sha256(), 0
            with archive.extractfile(member) as source, target.open("xb") as output:
                while chunk := source.read(min(1024 * 1024, item["bytes"] - count + 1)):
                    count += len(chunk)
                    if count > item["bytes"]:
                        raise ValueError("解包成員超過清單大小。")
                    digest.update(chunk)
                    output.write(chunk)
            if count != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(f"解包檔案 SHA-256 不符：{member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("壓縮包缺少清單中的檔案。")


def verify_directory(manifest, directory):
    directory = Path(directory).absolute()
    if directory.is_symlink() or not directory.is_dir() or any(parent.is_symlink() for parent in directory.parents):
        raise ValueError("資料目標必須是實際存在的目錄，不能經過連結。")
    expected = declarations(manifest["files"])
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("既有資料含連結或特殊檔案；請另選 --output。")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(expected):
        raise ValueError("既有資料缺少檔案或多出其他內容；不會覆蓋，請另選 --output。")
    for name, item in expected.items():
        path = directory / name
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"既有資料不同；不會覆蓋，請另選 --output：{name}")
    return directory


def fetch_data(
    manifest_path, output, *, revision=DEFAULT_REVISION, manifest_sha256=DEFAULT_MANIFEST_SHA256, opener=None
):
    digest_pin(revision, 40, "Git 版本")
    manifest = load_manifest(manifest_path, manifest_sha256)
    result = summary(manifest, revision, manifest_sha256)
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        verify_directory(manifest, destination)
        return {
            **result,
            "status": "existing_verified",
            "destination": str(destination),
            "files_verified": len(manifest["files"]),
        }
    if any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("資料目標不能經過符號連結。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".natural-data-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        data = temporary / "data"
        data.mkdir()
        manifest_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/docs/natural-assistant/manifest.json"
        manifest_record = download_checked(
            manifest_url,
            temporary / "manifest.json",
            Path(manifest_path).stat().st_size,
            manifest_sha256,
            opener=opener,
        )
        transport = []
        for specification in manifest["archives"]:
            archive = temporary / PurePosixPath(specification["path"]).name
            url = f"https://media.githubusercontent.com/media/{REPOSITORY}/{revision}/{specification['path']}"
            transport.append(
                download_checked(url, archive, specification["bytes"], specification["sha256"], opener=opener)
            )
            extract_checked(archive, specification, data)
        verify_directory(manifest, data)
        if destination.exists() or destination.is_symlink():
            raise ValueError("下載期間目標資料夾已出現；不會覆蓋，請另選 --output。")
        data.rename(destination)
    verify_directory(manifest, destination)
    return {
        **result,
        "status": "downloaded_verified",
        "destination": str(destination),
        "files_verified": len(manifest["files"]),
        "remote_manifest": manifest_record,
        "transport": transport,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--revision", default=DEFAULT_REVISION, help="固定的 40 位 Git commit；預設為已公開的三包資料版本"
    )
    parser.add_argument(
        "--manifest-sha256", default=DEFAULT_MANIFEST_SHA256, help="本機與該 Git 版本的資料清單都必須符合此固定 SHA-256"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/natural")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list", action="store_true")
    modes.add_argument("--verify", action="store_true", help="只核對既有資料，不下載")
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest, args.manifest_sha256)
        if args.list:
            result = summary(manifest, args.revision, args.manifest_sha256)
        elif args.verify:
            result = {
                **summary(manifest, args.revision, args.manifest_sha256),
                "status": "verified",
                "destination": str(verify_directory(manifest, args.output)),
                "files_verified": len(manifest["files"]),
            }
        else:
            result = fetch_data(
                args.manifest, args.output, revision=args.revision, manifest_sha256=args.manifest_sha256
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, tarfile.TarError, UnicodeError, EOFError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
