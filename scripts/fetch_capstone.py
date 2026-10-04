"""Download one published capstone stage; pin HF revision, verify SHA and run CPU inference."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capstone_release import (  # noqa: E402
    MANIFEST,
    file_sha256,
    safe_relative,
    validate_capstone_payload,
    validate_manifest,
)


def fetch_capstone(manifest, stage, output):
    import torch
    from huggingface_hub import hf_hub_download

    validate_manifest(manifest)
    selected = [model for model in manifest["models"] if model["id"] == stage]
    if len(selected) != 1:
        raise ValueError("Unknown published capstone stage; use --list")
    specification = selected[0]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / stage
    if target.is_symlink():
        raise ValueError("Download target cannot be a symlink")
    if target.exists():
        raise ValueError("Selected stage already exists; choose another --output to retain previous downloads")
    # Validate every download before moving a complete stage into its destination.
    with tempfile.TemporaryDirectory(prefix=".capstone-", dir=output) as temporary:
        directory = Path(temporary)
        for item in specification["files"]:
            cached = Path(hf_hub_download(manifest["repo"], item["path"], revision=manifest["revision"], token=False))
            if file_sha256(cached) != item["sha256"] or cached.stat().st_size != item["bytes"]:
                raise ValueError("Downloaded capstone bytes do not match the pinned public SHA/size")
            destination = directory / safe_relative(item["output"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, destination)
        checkpoint = directory / specification["checkpoint"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format_version") != specification["format_version"]:
            raise ValueError("Downloaded capstone format disagrees with its public manifest")
        bits = saved.get("quantization", {}).get("bits")
        actual_stage = f"{saved['stage']}-int{bits}" if bits else saved["stage"]
        if actual_stage != stage:
            raise ValueError("Downloaded checkpoint stage disagrees with its public manifest")
        validate_capstone_payload(saved)
        directory.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stage", help="Download one published stage shown by --list")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/capstone")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error("Capstone weights have not been published: no verified public manifest exists yet")
    try:
        manifest = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
        if args.list or not args.stage:
            for model in manifest["models"]:
                print(
                    json.dumps(
                        {
                            "stage": model["id"],
                            "repo": manifest["repo"],
                            "revision": manifest["revision"],
                            "format_version": model["format_version"],
                        },
                        ensure_ascii=False,
                    )
                )
        else:
            print(str(fetch_capstone(manifest, args.stage, args.output)))
    except (KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
