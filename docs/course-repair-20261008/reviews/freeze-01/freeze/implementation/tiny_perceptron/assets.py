"""資料快照的完整性檢查與解包；不覆蓋讀者已修改的資料。"""

import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack_asset(archive, asset, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    if archive.stat().st_size != asset["archive_bytes"] or sha256(archive) != asset["archive_sha256"]:
        raise ValueError("資料包大小／SHA-256 不符，或目前只有 Git LFS pointer")
    expected = {row["path"]: row for row in asset["files"]}
    if len(expected) != len(asset["files"]) or not expected:
        raise ValueError("manifest 的檔名重複或沒有內容")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tiny-training-", dir=destination.parent) as directory:
        temporary = Path(directory)
        seen = set()
        with tarfile.open(archive, "r:gz") as package:
            for member in package:
                path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.name
                    or ":" in member.name
                    or member.name in seen
                ):
                    raise ValueError("資料包含不允許的路徑、連結或重複檔案")
                row = expected.get(member.name)
                if row is None or member.size != row["bytes"]:
                    raise ValueError("資料包內容不符合 manifest")
                target = temporary.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if sha256(target) != row["sha256"]:
                    raise ValueError(f"資料檔案 SHA-256 不符：{member.name}")
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError("資料包缺少 manifest 中的檔案")
        # 全部核對後才寫入，先確認既有資料不會被改寫。
        for name, row in expected.items():
            target = destination / name
            if not target.resolve().is_relative_to(destination) or target.is_symlink():
                raise ValueError(f"目標路徑不在資料目錄內：{name}")
            if target.exists() and (not target.is_file() or sha256(target) != row["sha256"]):
                raise ValueError(f"既有資料不同，請另選 --output：{name}")
        written = 0
        for name in expected:
            target = destination / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(temporary / name, target)
                written += 1
    return {
        "asset": asset["id"],
        "files_verified": len(expected),
        "files_written": written,
        "destination": str(destination),
    }
