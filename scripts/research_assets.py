"""固定研究明細的按需下載、離線核對與原始內容發布；不執行訓練。"""

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CATALOG = "assets/research/catalog.json"
REPOSITORY = "birdhackor/tiny-perceptron-vlm"
SOURCE_REVISION = "92b8763ca24511f0da1c259429851a314accea19"
RELEASE_TAG = "research-inputs-20261102"
LICENSE_FILES = {
    "natural-v4": {
        "docs/natural-assistant-v4/research/chat-speech.Apache-2.0-LICENSE",
        "docs/natural-assistant-v4/research/chat-speech.NOTICE.txt",
    },
    "selftrained-sources": {
        "docs/selftrained/DATA-NOTICE.md",
        "docs/selftrained/V2-DATA-NOTICE.md",
        "docs/selftrained/V2-VOICE-NOTICE.md",
    },
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"研究資料路徑不合法：{name}")
    target = root / name
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError(f"研究資料路徑離開專案：{name}")
    return target


def parse_catalog(raw):
    data = json.loads(raw)
    if (
        data.get("schema_version") != 1
        or data.get("repository") != REPOSITORY
        or data.get("source_revision") != SOURCE_REVISION
        or data.get("release_tag") != RELEASE_TAG
        or set(data.get("groups", {})) != {"natural-v4", "selftrained-sources"}
    ):
        raise ValueError("研究資料清單版本／來源不符")
    for group, spec in data["groups"].items():
        if PurePosixPath(spec["name"]).name != spec["name"] or "\\" in spec["name"]:
            raise ValueError("研究資料包名稱不合法")
        for name, item in spec["files"].items():
            allowed = (
                name == "docs/natural-assistant/v4/manifest.json" or name.startswith("docs/natural-assistant/v4/data/")
                if group == "natural-v4"
                else name in {"docs/selftrained/package-recipe.json", "docs/selftrained/v2-package-recipe.json"}
                or name.startswith("docs/selftrained/provenance/")
            )
            allowed = allowed or name in LICENSE_FILES[group]
            safe_path(ROOT, name)
            if (
                not allowed
                or type(item["bytes"]) is not int
                or item["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            ):
                raise ValueError(f"不是允許的固定研究輸入：{name}")
        if "url" in spec:
            expected = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{spec['name']}"
            if (
                spec["url"] != expected
                or type(spec["bytes"]) is not int
                or spec["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])
            ):
                raise ValueError("研究資料包的公開 URL／大小／SHA 不合法")
    return data


def load_catalog(root=ROOT):
    return parse_catalog((root / CATALOG).read_bytes())


def checked(raw, item, name):
    if len(raw) != item["bytes"] or sha(raw) != item["sha256"]:
        raise ValueError(f"研究資料大小／SHA-256 不符：{name}")
    return raw


def selected(catalog, group):
    if group == "all":
        return catalog["groups"]
    if group not in catalog["groups"]:
        raise ValueError(f"未知研究資料組：{group}")
    return {group: catalog["groups"][group]}


def verify(root=ROOT, group="all", catalog=None):
    catalog = catalog or load_catalog(root)
    for spec in selected(catalog, group).values():
        for name, item in spec["files"].items():
            path = safe_path(root, name)
            if not path.is_file():
                raise ValueError(f"缺少 {name}；先執行 python scripts/research_assets.py fetch --group {group}")
            checked(path.read_bytes(), item, name)


def unpack(raw, spec, root):
    """先驗完所有項目再寫檔；不接受多餘項目、symlink 或任意覆寫。"""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) != len(spec["files"]) or {entry.filename for entry in entries} != set(spec["files"]):
            raise ValueError("研究資料包的完整檔案集合不符")
        verified = {}
        for entry in entries:
            name = entry.filename
            target = safe_path(root, name)
            item = spec["files"][name]
            if stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size != item["bytes"]:
                raise ValueError(f"研究資料 ZIP 項目不符：{name}")
            verified[name] = checked(archive.read(entry), item, name)
            if target.exists():
                checked(target.read_bytes(), item, name)
        for name, content in verified.items():
            target = safe_path(root, name)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            try:
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def fetch(root=ROOT, group="all", opener=urllib.request.urlopen):
    catalog = load_catalog(root)
    for label, spec in selected(catalog, group).items():
        missing = False
        for name, item in spec["files"].items():
            path = safe_path(root, name)
            if path.exists():
                checked(path.read_bytes(), item, name)
            else:
                missing = True
        if not missing:
            print(f"已核對本機研究資料：{label}")
            continue
        if not all(key in spec for key in ("url", "bytes", "sha256")):
            raise ValueError("研究資料包尚未完成公開發布，不能下載未固定的候選內容")
        cache = root / "outputs/research-assets" / spec["name"]
        raw = cache.read_bytes() if cache.is_file() else b""
        if len(raw) != spec["bytes"] or sha(raw) != spec["sha256"]:
            with opener(spec["url"], timeout=120) as response:
                raw = response.read(spec["bytes"] + 1)
            checked(raw, spec, spec["name"])
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
        unpack(raw, spec, root)
        verify(root, label, catalog)
        print(f"已取得並核對研究資料：{label}")


def frozen_input(root, relative, revision):
    """只供已列名的研究輸入使用；清單必須來自當次程式的 Git 提交。

    付費授權、控制設定及其他 JSON 仍使用原本的當次提交核對。
    本函式不下載，不把缺少 committed JSON 當成可以相信任意本機檔案。
    """
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("研究輸入必須綁定完整的程式提交 SHA")
    raw_catalog = subprocess.check_output(["git", "show", f"{revision}:{CATALOG}"], cwd=root)
    catalog = parse_catalog(raw_catalog)
    for spec in catalog["groups"].values():
        if relative in spec["files"]:
            path = safe_path(root, relative)
            if not path.is_file():
                raise ValueError(
                    f"缺少固定研究輸入；先執行 python scripts/research_assets.py fetch --group all：{relative}"
                )
            raw = path.read_bytes()
            return checked(raw, spec["files"][relative], relative)
    raise ValueError(f"不是已列名的固定研究輸入：{relative}")


def build(root=ROOT, source_root=None):
    catalog = load_catalog(root)
    source_root = source_root or root
    verify(source_root, catalog=catalog)
    output = root / "outputs/research-assets"
    output.mkdir(parents=True, exist_ok=True)
    for spec in catalog["groups"].values():
        path = output / spec["name"]
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, item in sorted(spec["files"].items()):
                raw = checked(safe_path(source_root, name).read_bytes(), item, name)
                entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        raw = path.read_bytes()
        spec.update(bytes=len(raw), sha256=sha(raw))
        spec["url"] = f"https://github.com/{REPOSITORY}/releases/download/{catalog['release_tag']}/{spec['name']}"
    return catalog


def gh(*args):
    result = subprocess.run(["gh", *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("GitHub 研究資料發布操作失敗；沒有覆寫既有附件")
    return result.stdout


def release_by_tag(tag):
    result = subprocess.run(
        ["gh", "api", "--include", f"repos/{REPOSITORY}/releases/tags/{tag}"], capture_output=True, check=False
    )
    header, separator, body = result.stdout.partition(b"\r\n\r\n")
    if not separator:
        header, separator, body = result.stdout.partition(b"\n\n")
    status = re.match(rb"HTTP/\S+ (\d{3})", header)
    if not separator or status is None:
        raise RuntimeError("Release 查詢沒有可核對的 HTTP 狀態")
    if status[1] == b"404":
        return None
    if status[1] != b"200" or result.returncode:
        raise RuntimeError("Release 查詢失敗，不將權限或服務錯誤當成不存在")
    return json.loads(body)


def publish(root=ROOT):
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
    ):
        raise ValueError("公開發布限指定 repository 的 GitHub Actions；本機可 build 或 fetch")
    source_root = root / "outputs/research-source"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("發布來源不是固定的原始 Git 版本")
    catalog = build(root, source_root)
    tag = catalog["release_tag"]
    release = release_by_tag(tag)
    if release is None:
        gh(
            "release",
            "create",
            tag,
            "--target",
            catalog["source_revision"],
            "--latest=false",
            "--title",
            "Fixed research inputs: manifests, sources and annotations",
            "--notes",
            "固定研究來源、標註與重建配方；原始檔案 bytes 不變，不含模型權重，不執行訓練。",
        )
    release = json.loads(gh("api", f"repos/{REPOSITORY}/releases/tags/{tag}"))
    ref = json.loads(gh("api", f"repos/{REPOSITORY}/git/ref/tags/{tag}"))
    if release["draft"] or ref["object"].get("sha") != SOURCE_REVISION or ref["object"].get("type") != "commit":
        raise ValueError("研究資料 Release／tag 不是固定原始版本；不移動既有 tag")
    for spec in catalog["groups"].values():
        release = json.loads(gh("api", f"repos/{REPOSITORY}/releases/tags/{tag}"))
        matches = [asset for asset in release["assets"] if asset["name"] == spec["name"]]
        if not matches:
            gh("release", "upload", tag, str(root / "outputs/research-assets" / spec["name"]))
        # 無 Authorization 的公開下載同時驗證可取得性與全部原始檔案。
        with urllib.request.urlopen(spec["url"], timeout=120) as response:
            raw = response.read(64 * 1024 * 1024 + 1)
        with tempfile.TemporaryDirectory() as directory:
            unpack(raw, spec, Path(directory))
        # 既有附件可能來自另一個 zlib 版本；固定其實際壓縮包，絕不覆寫。
        spec.update(bytes=len(raw), sha256=sha(raw))
        release = json.loads(gh("api", f"repos/{REPOSITORY}/releases/tags/{tag}"))
        spec["asset_id"] = next(asset["id"] for asset in release["assets"] if asset["name"] == spec["name"])
    path = root / "outputs/research-assets/publication.json"
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    release = json.loads(gh("api", f"repos/{REPOSITORY}/releases/tags/{tag}"))
    if any(asset["name"] == path.name for asset in release["assets"]):
        with urllib.request.urlopen(
            f"https://github.com/{REPOSITORY}/releases/download/{tag}/{path.name}", timeout=120
        ) as response:
            if json.load(response) != catalog:
                raise ValueError("既有公開清單不同，不覆寫")
    else:
        gh("release", "upload", tag, str(path))
    print(
        json.dumps(
            {
                "release": tag,
                "groups": {
                    key: {field: value[field] for field in ("bytes", "sha256", "asset_id")}
                    for key, value in catalog["groups"].items()
                },
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "verify", "build", "publish"])
    parser.add_argument("--group", default="all")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(group=args.group)
    elif args.command == "verify":
        verify(group=args.group)
        print("固定研究資料離線核對通過")
    elif args.command == "build":
        print(
            json.dumps(
                {
                    key: {field: spec[field] for field in ("name", "bytes", "sha256")}
                    for key, spec in build()["groups"].items()
                },
                indent=2,
            )
        )
    else:
        publish()


if __name__ == "__main__":
    main()
