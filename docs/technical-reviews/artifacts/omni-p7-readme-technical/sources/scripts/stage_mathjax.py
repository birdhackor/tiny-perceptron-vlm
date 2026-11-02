"""核對固定 MathJax 資產，將公式排版所需檔案隨網站發布。第三方 bytes 不進 Git。"""

import hashlib
import json
import re
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "course/web/mathjax-assets.json"


def validate_manifest(value):
    if (
        value.get("schema_version") != 1
        or value.get("name") != "MathJax"
        or value.get("version") != "3.2.2"
        or value.get("repository") != "mathjax/MathJax"
        or value.get("license") != "Apache-2.0"
        or not re.fullmatch(r"[0-9a-f]{40}", value.get("revision", ""))
    ):
        raise ValueError("MathJax 資產需固定版本、來源與授權")
    files = value.get("files", [])
    if not 3 <= len(files) <= 150:
        raise ValueError("MathJax 資產清單大小不符")
    seen = set()
    total = 0
    for item in files:
        name = item["path"]
        path = PurePosixPath(name)
        if (
            not name
            or not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or path.as_posix() != name
        ):
            raise ValueError("MathJax 資產必須使用安全相對路徑")
        if name in seen or item["upstream_path"] != (name if name == "LICENSE" else "es5/" + name):
            raise ValueError("MathJax 資產路徑重複或與來源不符")
        if type(item.get("bytes")) is not int or not 0 < item["bytes"] <= 2 * 1024 * 1024:
            raise ValueError("MathJax 單檔大小不符")
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError("MathJax 缺少固定 SHA-256")
        seen.add(name)
        total += item["bytes"]
    required = {"LICENSE", "tex-chtml.js", "output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff"}
    if not required <= seen or total > 16 * 1024 * 1024:
        raise ValueError("MathJax 缺少必要資產或總量超限")
    return value


def verify(raw, item):
    if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("MathJax 資產指紋不符：" + item["path"])


def asset_path(base, name):
    """目的目錄自身、資產檔與中間目錄均不可透過 symlink 改換位置。"""
    base = Path(base).absolute()
    target = base / name
    for part in (target, *target.parents):
        if part.is_symlink():
            raise ValueError("MathJax 資產路徑不可使用 symlink")
        if part == base:
            break
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError("MathJax 資產路徑不可離開資產目錄")
    return target


def read_asset(path, item):
    with path.open("rb") as stream:
        raw = stream.read(item["bytes"] + 1)
    verify(raw, item)
    return raw


def prepare_assets(root=ROOT, manifest_path=None):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path) if manifest_path else root / "course/web/mathjax-assets.json"
    manifest = validate_manifest(json.loads(manifest_path.read_bytes()))
    cache = root / "outputs/mathjax-cache" / manifest["version"]

    def obtain(item):
        target = asset_path(cache, item["path"])
        if target.exists():
            read_asset(target, item)
            return
        url = "https://raw.githubusercontent.com/mathjax/MathJax/" + manifest["revision"] + "/" + item["upstream_path"]
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read(item["bytes"] + 1)
        verify(raw, item)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
        try:
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(obtain, manifest["files"]))
    return cache, manifest


def stage_assets(destination, prepared):
    cache, manifest = prepared
    validate_manifest(manifest)
    destination = Path(destination)
    # 在第一次寫入之前先檢查整份目的清單；不是只檢查 manifest 字串。
    for item in manifest["files"]:
        asset_path(destination, item["path"])
    for item in manifest["files"]:
        source = asset_path(cache, item["path"])
        raw = read_asset(source, item)
        target = asset_path(destination, item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
