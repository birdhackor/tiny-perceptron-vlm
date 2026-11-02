"""下載固定的 C-v3 配套，或把公開裝置請求交給已驗收的神經模型。"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "docs/decision-product/v3/public-release.json"
BUNDLE_FILES = {"bundle.json", "policy.json", "tokenizer.json", "weights.safetensors"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_bundle(directory, descriptor):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("配套必須是本機目錄")
    for name, expected in descriptor["bundle"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"C-v3 配套的內容指紋不符：{name}")


def install_archive(archive, output, descriptor):
    """只安裝四個經 repo 清單核對的推論檔，不執行封包內的程式。"""
    archive, output = Path(archive), Path(output)
    if archive.stat().st_size != descriptor["bytes"] or sha256(archive.read_bytes()) != descriptor["sha256"]:
        raise ValueError("下載封包的大小或 SHA-256 不符")
    if set(descriptor["bundle"]) != BUNDLE_FILES:
        raise ValueError("版本清單須明列四個推論檔")
    payload = {}
    prefix = "tiny-perceptron-c-v3/bundle/"
    with zipfile.ZipFile(archive) as source:
        for name, expected in descriptor["bundle"].items():
            member = prefix + name
            if source.namelist().count(member) != 1 or source.getinfo(member).file_size > 4 * 1024**2:
                raise ValueError("封包的推論檔重複、缺少或過大")
            data = source.read(member)
            if sha256(data) != expected:
                raise ValueError(f"封包內的內容指紋不符：{name}")
            payload[name] = data
    if output.exists() or output.is_symlink():
        verify_bundle(output, descriptor)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".c-v3-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        for name, data in payload.items():
            (bundle / name).write_bytes(data)
        # 不覆寫既有版本；只有四個檔全部核對成功後才安裝。
        if output.exists() or output.is_symlink():
            raise FileExistsError("請選擇新的 C-v3 配套路徑")
        bundle.rename(output)


def download(output, descriptor):
    with tempfile.TemporaryDirectory(prefix="course-c-v3-") as temporary:
        archive = Path(temporary) / "release.zip"
        request = urllib.request.Request(descriptor["url"], headers={"User-Agent": "tiny-perceptron-course"})
        with urllib.request.urlopen(request, timeout=90) as response, archive.open("xb") as target:
            shutil.copyfileobj(_LimitedReader(response, descriptor["bytes"] + 1), target)
        install_archive(archive, output, descriptor)


class _LimitedReader:
    """超過固定下載大小就停止，交由封包檢查回報。"""

    def __init__(self, source, remaining):
        self.source, self.remaining = source, remaining

    def read(self, size):
        data = self.source.read(min(size, self.remaining)) if self.remaining else b""
        self.remaining -= len(data)
        return data


def dispatch_request(engine, request):
    # devices 是使用者提供的公開格式標記；這裡沒有讀正解或計算動作規則。
    if not isinstance(request, dict) or "devices" not in request:
        raise ValueError("本入口支援 devices 裝置請求；文字 Auto 與 Looped 請使用各自的既有入口")
    from tiny_perceptron.decision_product.cli import predict_request

    return {"product": "C-v3", "dispatch": "typed-devices", **predict_request(engine, request)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="匿名取得並核對固定版本")
    get.add_argument("--output", type=Path, default=ROOT / "checkpoints/decision-product-v3")
    judge = commands.add_parser("judge", help="以神經模型回覆裝置請求")
    judge.add_argument("--bundle", type=Path, default=ROOT / "checkpoints/decision-product-v3")
    judge.add_argument("--request", type=Path, default=ROOT / "examples/decision_product/record-ready.json")
    args = parser.parse_args()
    descriptor = json.loads(RELEASE.read_text(encoding="utf-8"))
    try:
        if args.command == "download":
            download(args.output, descriptor)
            result = {"product": "C-v3", "bundle": str(args.output), "verified": True}
        else:
            import torch

            from tiny_perceptron.decision_product.service import DecisionEngine

            verify_bundle(args.bundle, descriptor)
            torch.set_num_threads(2)
            result = dispatch_request(DecisionEngine(args.bundle), json.loads(args.request.read_text(encoding="utf-8")))
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(2, f"無法完成操作：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
