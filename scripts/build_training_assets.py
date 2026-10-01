"""把首批已收集資料製作成可重建的分來源快照；不下載、不訓練、不上傳。"""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from tiny_perceptron.assets import sha256

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def definitions(source):
    text = load(source / "text-initial/asset-manifest.json")
    tiny, poetry = text["assets"]
    result = [
        ("tinystories", "text-initial", 512, tiny["license"], tiny, "tinystories-*", "cdla-sharing-*"),
        ("chinese-poetry", "text-initial", 365, poetry["license"], poetry, "chinese-*", "tang300-source.json"),
    ]
    for name in ("ultrachat-sft", "ultrafeedback-dpo", "pku-safe-rlhf"):
        directory = "behavior-initial/" + name
        info = load(source / directory / "asset.json")
        result.append((name, directory, 100, info["license"], info, "**/*"))
    for name, directory, count, metadata in (
        ("fashion-mnist", "vision-initial", 50, "manifest.json"),
        ("gsm8k", "reasoning-initial", 200, "asset-manifest.json"),
        ("fsdd", "fsdd-initial", 20, "asset.json"),
    ):
        info = load(source / directory / metadata)
        license_id = info["license"]
        if isinstance(license_id, dict):
            license_id = license_id["identifier"]
        result.append((name, directory, count, license_id, info, "**/*"))
    return result


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "sources").mkdir(exist_ok=True)
    assets = []
    for name, directory, count, license_id, info, *patterns in definitions(source):
        location = source / directory
        paths = {path for pattern in patterns for path in location.glob(pattern) if path.is_file()}
        if name in ("tinystories", "chinese-poetry"):
            paths.update((location / "README.md", location / "asset-manifest.json"))
        entries = {path.relative_to(source).as_posix(): path.read_bytes() for path in sorted(paths)}
        notice_path = f"{directory}/{name}-LICENSE-AND-SOURCE.md"
        notice = (
            f"# {name} 教學資料快照 v1\n\n"
            f"資料授權：**{license_id}**；本 repo 的 MIT 程式授權不改變資料授權。\n\n"
            "保留上游註記、來源版本與原始授權文件；選樣、轉換與用途見包內 README／manifest，"
            f"以及 Git 中的 assets/training/sources/{name}.json。此包僅為 pilot，沒有模型訓練結果。\n"
        )
        entries[notice_path] = notice.encode("utf-8")
        archive = output / f"{name}-v1.tar.gz"
        with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as package:
                for filename, content in sorted(entries.items()):
                    member = tarfile.TarInfo(filename)
                    member.size, member.mode, member.mtime = len(content), 0o644, 0
                    package.addfile(member, io.BytesIO(content))
        (output / "sources" / f"{name}.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        archive_path = f"assets/training/{archive.name}"
        asset = {
            "id": name,
            "version": 1,
            "archive": archive_path,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256(archive),
            "license": license_id,
            "training_records": count,
            "source_metadata": f"assets/training/sources/{name}.json",
            "target": directory,
            "files": [
                {"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                for path, content in sorted(entries.items())
            ],
        }
        assets.append(asset)
        print(name, asset["archive_bytes"], "bytes", license_id)
    manifest = {
        "schema_version": 1,
        "release": "training-pilot-v1",
        "assets": assets,
        "training_records_prepared": sum(asset["training_records"] for asset in assets),
        "scope": "Collected pilot subsets; holdout design still required. FSDD includes validation/test companions; Fashion-MNIST also retains source train parquet. No trained-model quality claim.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/training")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/training")
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
