"""發布包的純離線核對；不需要讀HF token或載入Torch。"""

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack_verified(archive, expected_sha, destination):
    if sha(archive) != expected_sha:
        raise ValueError("發布transport不符已提交SHA")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("發布展開目錄必須全新；不覆寫先前結果")
    destination.mkdir(parents=True)
    seen, total = set(), 0
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or member.name in seen
            ):
                raise ValueError("發布包只允許唯一相對路徑的一般檔案")
            total += member.size
            if total > 64 * 1024 * 1024 or len(seen) >= 100:
                raise ValueError("發布包超過固定64MiB／100檔上限")
            seen.add(member.name)
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            raw = source.read(member.size + 1)
            if len(raw) != member.size:
                raise ValueError("發布包member長度不符")
            path.write_bytes(raw)
    return destination


def verify_release(directory, spec):
    directory = Path(directory)
    expected = spec["files"]
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(expected):
        raise ValueError("發布檔案集合不符已提交清單")
    for name, item in expected.items():
        path = directory / name
        if path.is_symlink() or sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("發布檔案SHA／大小不符：" + name)
    freeze_path = directory / "settings-freeze.json"
    if sha(directory / "data-manifest.json") != spec["data_manifest_sha256"]:
        raise ValueError("公開資料manifest不同於訓練／最後測試版本")
    freeze = json.loads(freeze_path.read_bytes())
    report = json.loads((directory / "report.json").read_bytes())
    if (
        report["settings_freeze_sha256"] != sha(freeze_path)
        or report["data_manifest_sha256"] != spec["data_manifest_sha256"]
        or freeze["data_manifest_sha256"] != spec["data_manifest_sha256"]
    ):
        raise ValueError("正式結果沒有綁同一模型設定／資料凍結")
    if any(report["gates"].get(key) is not True for key in ("b_primary", "c_choice", "c_reorder")):
        raise ValueError("B/C必要成品門檻未過，不可發布為已驗收成品")
    for product in ("b", "c"):
        manifest = json.loads((directory / product / "artifact-manifest.json").read_bytes())
        if (
            manifest.get("product") != product
            or manifest.get("settings_freeze_sha256") != sha(freeze_path)
            or manifest.get("data_manifest_sha256") != spec["data_manifest_sha256"]
        ):
            raise ValueError("公開artifact manifest凍結身分不符")
        for name, item in manifest["files"].items():
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("成品檔名不可跳出自己的目錄")
            path = directory / product / name
            if sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("公開成品檔案不同")
            if name in freeze["products"][product] and sha(path) != freeze["products"][product][name]:
                raise ValueError("公開成品不同於最後測試凍結")
        if not {
            "model.pt",
            "config.json",
            "tokenizer.json",
            "training-receipt.json",
            "calibration.json" if product == "c" else "auto-policy.json",
        } <= set(manifest["files"]):
            raise ValueError("公開成品缺必要權重或推論設定")
    return {
        "files": len(actual),
        "settings_freeze_sha256": sha(freeze_path),
        "gates": report["gates"],
        "scope": report["scope"],
    }
