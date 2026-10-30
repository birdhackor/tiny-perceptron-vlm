"""匿名下載已發布的固定微型產物；只驗資料，不反序列化完整續訓狀態。"""

import argparse
import hashlib
import json
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "birdhackor/tiny-perceptron-vlm"
MAX_ARCHIVE = 128 * 1024**2
MAX_EXPANDED = 256 * 1024**2
KINDS = {"av", "choice", "choice_control", "codec", "talker"}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    value = json.loads(Path(path).read_bytes())
    if (
        value.get("schema") != "omni-public-model-download-v1"
        or value["repository"] != REPOSITORY
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}", value["release_tag"]) is None
    ):
        raise ValueError("公開產物版本描述不合法")
    reference = value["model_selection"]
    if (
        reference["path"] != "docs/omni-products/v1/model-selection.json"
        or file_sha(ROOT / reference["path"]) != reference["sha256"]
    ):
        raise ValueError("公開產物與已提交model selection不符")
    selected = json.loads((ROOT / reference["path"]).read_bytes())
    if (
        selected.get("schema") != "omni-core-model-selection-v1"
        or set(selected["selected"]) != KINDS
        or set(selected["model_files"]) != KINDS
        or set(value["components"]) != KINDS
    ):
        raise ValueError("公開產物缺少選定模組")
    for kind, row in value["components"].items():
        if (
            row["job"] != selected["selected"][kind]
            or row["name"] != row["job"] + ".zip"
            or re.fullmatch(r"[a-z0-9-]+", row["job"]) is None
            or type(row["bytes"]) is not int
            or not 0 < row["bytes"] <= MAX_ARCHIVE
            or re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None
        ):
            raise ValueError("公開成品下載與固定候選不同")
        selected_files(selected, kind)
    return {**value, "_model_selection": selected}


def selected_files(selected, kind):
    """三份身分來自已提交selection，不採信ZIP自己的manifest作唯一依據。"""
    job = selected["selected"][kind]
    binding = selected["model_files"][kind]
    if re.fullmatch(r"[a-z0-9-]+", job) is None or binding["job"] != job:
        raise ValueError("選定模組的工作身分不同")
    prefix = f"core/{job}/"
    expected = {
        prefix + "training/weights.safetensors": binding["weights"],
        prefix + "training/training-result.json": {"sha256": binding["training_result_sha256"]},
        prefix + "job-result.json": {"sha256": selected["all_formal_jobs"][job]["job_result_sha256"]},
    }
    for row in expected.values():
        if re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None:
            raise ValueError("選定模組缺完整SHA")
        if "bytes" in row and (type(row["bytes"]) is not int or not 0 < row["bytes"] <= MAX_EXPANDED):
            raise ValueError("選定模組的權重大小不同")
    return expected


def verify_installed(destination, selected, kind):
    for name, binding in selected_files(selected, kind).items():
        target = Path(destination).joinpath(*PurePosixPath(name).parts)
        if (
            target.is_symlink()
            or not target.is_file()
            or file_sha(target) != binding["sha256"]
            or ("bytes" in binding and target.stat().st_size != binding["bytes"])
        ):
            raise ValueError("本機權重或訓練收據不是已選定成品")


def safe_target(destination, name):
    """檢查下載根目錄以內的既有路徑；不跟隨檔案或中間目錄symlink。"""
    destination = Path(destination).absolute()
    target = destination.joinpath(*PurePosixPath(name).parts)
    for parent in [target, *target.parents]:
        if parent == destination.parent:
            break
        if parent.is_symlink():
            raise ValueError("不透過symlink寫入已存在位置")
        if parent != target and parent.exists() and not parent.is_dir():
            raise ValueError("安裝路徑的既有父節點不是目錄")
    return target


def portable_member(filename):
    name = PurePosixPath(filename)
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)]}
    return (
        not name.is_absolute()
        and name.as_posix() == filename
        and all(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part)
            and not part.endswith(".")
            and part.split(".")[0].upper() not in reserved
            for part in name.parts
        )
    )


def fetch(value, asset, target):
    url = f"https://github.com/{REPOSITORY}/releases/download/{urllib.parse.quote(value['release_tag'], safe='')}/{urllib.parse.quote(asset['name'], safe='')}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response, Path(target).open("xb") as stream:
            if urllib.parse.urlparse(response.url).scheme != "https":
                raise ValueError("公開下載拒絕降為非HTTPS")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != asset["bytes"]:
                raise ValueError("公開下載長度不同")
            received = 0
            while block := response.read(1024**2):
                received += len(block)
                if received > asset["bytes"]:
                    raise ValueError("公開下載超出固定bytes上界")
                stream.write(block)
    except (urllib.error.URLError, OSError):
        # 不把GitHub臨時redirect的查詢參數帶進使用者log。
        raise RuntimeError("公開模型下載失敗；請確認版本已發布與網路連線") from None
    if Path(target).stat().st_size != asset["bytes"] or file_sha(target) != asset["sha256"]:
        raise ValueError("公開下載SHA／bytes與固定成品不同")


def install(archive_path, asset, destination, full_state=False, *, expected_files):
    destination = Path(destination)
    if file_sha(archive_path) != asset["sha256"] or Path(archive_path).stat().st_size != asset["bytes"]:
        raise ValueError("未核對的archive不能解包")
    prefix = "core/" + asset["job"] + "/"
    installed = {}
    with zipfile.ZipFile(archive_path) as archive:
        files = archive.infolist()
        names = {row.filename for row in files}
        if (
            len(names) != len(files)
            or len({name.casefold() for name in names}) != len(files)
            or len(files) > 64
            or sum(row.file_size for row in files) > MAX_EXPANDED
        ):
            raise ValueError("成品展開超界或重複路徑")
        for row in files:
            if (
                not portable_member(row.filename)
                or (row.external_attr >> 16) & 0o170000 == 0o120000
                or (row.filename != "package-file-manifest.json" and not row.filename.startswith(prefix))
            ):
                raise ValueError("成品含不允許的路徑")
        manifest = json.loads(archive.read("package-file-manifest.json"))
        if set(manifest) != names - {"package-file-manifest.json"}:
            raise ValueError("成品內manifest缺漏")
        required = {
            prefix + "training/weights.safetensors",
            prefix + "training/training-result.json",
            prefix + "job-result.json",
        }
        if not required <= manifest.keys():
            raise ValueError("成品缺推論weights／原始訓練收據")
        if set(expected_files) != required:
            raise ValueError("成品必須綁定已選weights與兩份訓練收據")
        for name, binding in expected_files.items():
            if any(manifest[name].get(key) != value for key, value in binding.items()):
                raise ValueError("成品內權重或訓練收據與已選定SHA不同")
        payload = {}
        for name, binding in manifest.items():
            raw = archive.read(name)
            if len(raw) != binding["bytes"] or hashlib.sha256(raw).hexdigest() != binding["sha256"]:
                raise ValueError("成品內逐檔SHA／bytes不同")
            if not full_state and name.endswith(".pt"):
                continue
            target = safe_target(destination, name)
            if target.exists():
                if target.stat().st_size != len(raw) or file_sha(target) != binding["sha256"]:
                    raise ValueError("既有產物不同；請選新目錄，不覆寫使用者檔案")
            payload[name] = raw
            installed[name] = binding
        # 全部SHA、已選身分與既有檔先核對，才開始安裝，避免晚發現的壞檔留下前面權重。
        for name, raw in payload.items():
            target = safe_target(destination, name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(raw)
    return installed


def download(manifest, destination, include_control=False, full_state=False):
    value = descriptor(manifest)
    receipt = safe_target(destination, "download-receipt.json")
    result = {
        "schema": "omni-public-model-download-receipt-v1",
        "descriptor_sha256": file_sha(manifest),
        "release_tag": value["release_tag"],
        "anonymous": True,
        "full_continuation_requested": full_state,
        "assets": {},
    }
    for kind, asset in value["components"].items():
        if kind == "choice_control" and not include_control:
            continue
        with tempfile.TemporaryDirectory(prefix="omni-public-download-") as temporary:
            target = Path(temporary) / asset["name"]
            fetch(value, asset, target)
            files = install(
                target,
                asset,
                destination,
                full_state,
                expected_files=selected_files(value["_model_selection"], kind),
            )
        result["assets"][kind] = {"job": asset["job"], "archive_sha256": asset["sha256"], "files": files}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = safe_target(destination, "download-receipt.json")
    if receipt.exists() and json.loads(receipt.read_bytes()) != result:
        raise ValueError("既有下載收據不同，請選新目錄")
    if not receipt.exists():
        with receipt.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/omni-products/v1/public-release.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-control", action="store_true")
    parser.add_argument("--full-checkpoints", action="store_true")
    args = parser.parse_args()
    result = download(args.manifest, args.output, args.include_control, args.full_checkpoints)
    print(
        json.dumps(
            {"output": str(args.output), "components": list(result["assets"]), "anonymous": True}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
