"""列出或按需取得已發布的 LFS 訓練資料，解包至 ignored data/training。"""

import argparse
import json
import subprocess
from pathlib import Path

from tiny_perceptron.assets import unpack_asset

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--asset", action="append", help="可重複指定；--asset all 取全部")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets/training/manifest.json").read_text(encoding="utf-8"))
    assets = {asset["id"]: asset for asset in manifest["assets"]}
    if args.list or not args.asset:
        for asset in assets.values():
            print(
                json.dumps(
                    {key: asset[key] for key in ("id", "training_records", "archive_bytes", "license")},
                    ensure_ascii=False,
                )
            )
        return
    selected = list(assets) if args.asset == ["all"] else list(dict.fromkeys(args.asset))
    if any(name not in assets for name in selected):
        parser.error("未知的 asset；先用 --list 查看，不會自動下載其他資料")
    for name in selected:
        asset = assets[name]
        archive = ROOT / asset["archive"]
        with archive.open("rb") as source:
            pointer = source.read(128).startswith(b"version https://git-lfs.github.com/spec/v1\n")
        if pointer:
            subprocess.run(
                ["git", "lfs", "pull", "origin", "--include", asset["archive"], "--exclude", ""], cwd=ROOT, check=True
            )
        print(json.dumps(unpack_asset(archive, asset, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
