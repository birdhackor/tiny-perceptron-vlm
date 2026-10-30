"""固定舊A/B輸入的獨立下載程序；先設定Hub網路，再載入模型套件。"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "docs/omni-products/v1"
REPOSITORY = "birdhackor/tiny-perceptron-vlm"


def configure_download_network():
    if "scripts.omni_revision.formal_local" in sys.modules:
        raise ValueError("舊輸入下載必須使用獨立新程序，不能重設正式量測的離線狀態")
    os.environ.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    from huggingface_hub import constants

    if constants.HF_HUB_OFFLINE is not False:
        raise ValueError("Hub已先快取離線設定；請使用獨立新下載程序")
    return {"HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"], "hub_client_offline": constants.HF_HUB_OFFLINE}


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def legacy_inputs(output):
    """匿名固定舊資料與舊A/B；只下載／核SHA／load，不做預測。"""
    configure_download_network()
    from tiny_perceptron.assets import unpack_asset
    from tiny_perceptron.lesson_products.delivery import download_product
    from tiny_perceptron.omni_products.media_bindings import OLD_A, OLD_B
    from tiny_perceptron.omni_products.training_runtime import file_sha
    from tiny_perceptron.selftrained.inference import fetch_public_export

    manifest = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_bytes())
    definition = json.loads((FROZEN / "data-definition.json").read_bytes())
    if (
        file_sha(ROOT / "docs/selftrained/v2-manifest.json")
        != definition["files_sha256"]["docs/selftrained/v2-manifest.json"]
    ):
        raise ValueError("舊資料manifest未符合事先凍結")
    package = manifest["package"]
    if (
        package["kind"] != "git-lfs"
        or package["repo_id"] != REPOSITORY
        or re.fullmatch("[0-9a-f]{40}", package["revision"]) is None
        or package["path"] != "assets/training/selftrained-v2.tar.gz"
        or not 0 < package["bytes"] <= 80 * 1024**2
        or package["unpacked_bytes"] != sum(row["bytes"] for row in [*manifest["records"], *manifest["assets"]])
    ):
        raise ValueError("舊資料下載缺固定版本／有界套件")
    root = ROOT / "outputs/omni-final-legacy-v1"
    root.mkdir(parents=True, exist_ok=False)
    archive = root / "data.tar.gz"
    url = f"https://media.githubusercontent.com/media/{REPOSITORY}/{package['revision']}/{package['path']}"
    with urllib.request.urlopen(url, timeout=60) as response, archive.open("xb") as stream:
        received = 0
        while block := response.read(1024**2):
            received += len(block)
            if received > package["bytes"]:
                raise ValueError("舊資料下載超界")
            stream.write(block)
    unpack_asset(
        archive,
        {
            "id": "fixed-selftrained-v2",
            "archive_bytes": package["bytes"],
            "archive_sha256": package["sha256"],
            "files": [*manifest["records"], *manifest["assets"]],
        },
        root / "data",
    )
    archive.unlink()
    a = fetch_public_export(
        OLD_A["repo"], OLD_A["revision"], root / "a", prefix=OLD_A["prefix"], manifest_sha256=OLD_A["manifest_sha256"]
    )
    download_product(OLD_B["repo"], OLD_B["revision"], OLD_B["prefix"], OLD_B["manifest_sha256"], root / "b", "b")
    if "scripts.omni_revision.formal_local" in sys.modules:
        raise ValueError("下載程序意外混入正式量測模組")
    json_file(
        output / "legacy-download.json",
        {
            "stage": "fixed-anonymous-input-download-only",
            "data_files_verified": len(manifest["records"]) + len(manifest["assets"]),
            "data_bytes": package["unpacked_bytes"],
            "anonymous_A_download": a,
            "B_manifest_sha256": OLD_B["manifest_sha256"],
            "no_NN_predictions": True,
            "formal_local_not_imported": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-network-only", action="store_true")
    args = parser.parse_args()
    if args.check_network_only:
        print(json.dumps(configure_download_network()))
    else:
        if args.output is None:
            raise ValueError("下載需要明示新的輸出目錄")
        legacy_inputs(args.output)


if __name__ == "__main__":
    main()
