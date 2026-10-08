"""Fetch ten frozen official CC-OCR examples for external evaluation only.

The official Hugging Face release stores original image bytes as base64 TSV,
not ZIP. Download and verify the two complete pinned TSV files, decode only
the preselected rows, and keep images outside Git/LFS. Author GT stays intact.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checked_path(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError(f"Asset path escapes output: {relative}")
    return path


def download(spec, path):
    """Only accept the frozen byte count and hash, including cached sources."""
    expected_size, expected_hash = int(spec["bytes"]), spec["sha256"]
    if path.exists():
        if path.stat().st_size == expected_size and digest(path) == expected_hash:
            return path
        raise ValueError(f"Cached source checksum mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "tiny-perceptron-vlm-external-ocr/1"})
    result = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                if size > expected_size:
                    raise ValueError("Source exceeded frozen byte count")
                result.update(block)
                output.write(block)
        if size != expected_size or result.hexdigest() != expected_hash:
            raise ValueError(f"Downloaded source checksum mismatch: {spec['url']}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def prepare(metadata, output, cache_dir, longest_side=0):
    frozen = json.loads(metadata.read_text(encoding="utf-8"))
    if frozen.get("schema_version") != 1 or frozen.get("training_allowed") is not False:
        raise ValueError("This generator requires a frozen external-evaluation-only manifest")
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sources = {source["id"]: source for source in frozen["sources"]}
    expected_files = {item["path"]: item for item in frozen["files"]}
    wanted = {}
    for row in frozen["rows"]:
        if row["split"] != "test" or row["task"] != "ocr":
            raise ValueError("External images must remain test-only OCR, never training")
        source = row["source"]
        key = (source["id"], int(source["row_index"]))
        if key in wanted:
            raise ValueError("Duplicate official row selection")
        wanted[key] = row
        checked_path(output, row["image"])
    if len(wanted) != 10:
        raise ValueError("Frozen selection must contain all ten original examples")
    extracted, downloaded = [], []
    csv.field_size_limit(1024 * 1024 * 1024)
    for ident, spec in sources.items():
        if spec["format"] != "tsv-with-base64-images":
            raise ValueError("Only the official pinned TSV format is supported")
        cached = checked_path(cache_dir, spec["cache_file"])
        source_path = download(spec, cached)
        downloaded.append(
            {"id": ident, "url": spec["url"], "bytes": source_path.stat().st_size, "sha256": digest(source_path)}
        )
        with source_path.open(encoding="utf-8", newline="") as stream:
            for index, source_row in enumerate(csv.DictReader(stream, delimiter="\t")):
                row = wanted.get((ident, index))
                if row is None:
                    continue
                note = row["source"]
                if int(source_row["index"]) != index or source_row["image_name"] != note["image_name"]:
                    raise ValueError(f"Official row identity mismatch: {row['id']}")
                if source_row["answer"] != note["original_gt"] or source_row["answer"] != row["answer"]:
                    raise ValueError(f"Official GT mismatch: {row['id']}")
                if hashlib.sha256(source_row["answer"].encode()).hexdigest() != note["original_gt_sha256"]:
                    raise ValueError(f"Original GT checksum mismatch: {row['id']}")
                raw = base64.b64decode(source_row["image"], validate=True)
                expected = expected_files[row["image"]]
                if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                    raise ValueError(f"Original image checksum mismatch: {row['id']}")
                with Image.open(io.BytesIO(raw)) as image:
                    if list(image.size) != note["image_size"]:
                        raise ValueError(f"Original image dimensions mismatch: {row['id']}")
                    image.verify()
                path = checked_path(output, row["image"])
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.read_bytes() != raw:
                    raise ValueError(f"Preserving an existing different image: {path}")
                path.write_bytes(raw)
                extracted.append(row["id"])
    if set(extracted) != {row["id"] for row in frozen["rows"]}:
        raise ValueError("A selected official row was not decoded")
    for spec in frozen["source_documents"]:
        download(spec, checked_path(output, spec["path"]))
    if longest_side:
        resized_files = []
        for row in frozen["rows"]:
            original = checked_path(output, row["image"])
            with Image.open(original) as opened:
                image = opened.convert("RGB")
                image.thumbnail((longest_side, longest_side), resample=Image.Resampling.LANCZOS)
                relative = f"inputs/{row['id']}-{longest_side}px.png"
                path = checked_path(output, relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                row["image"] = relative
                row["input_image_size"] = list(image.size)
                resized_files.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
        frozen["files"] = resized_files
        frozen["dataset_version"] += f"-input{longest_side}px"
        frozen["input_resize"] = {
            "longest_side": longest_side,
            "resampler": "Pillow Lanczos, preserve aspect ratio, no enlargement",
            "originals": "verified original bytes remain in images/; files/rows point to disclosed resized inputs",
        }
    manifest = output / "manifest.json"
    if longest_side:
        manifest.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        manifest.write_bytes(metadata.read_bytes())
    receipt = {
        "status": "passed",
        "source_metadata_sha256": digest(metadata),
        "manifest_sha256": digest(manifest),
        "downloaded_sources": downloaded,
        "selected_original_rows": extracted,
        "original_image_count": len(extracted),
        "full_tsv_hashes_verified": True,
        "all_original_gt_and_image_hashes_verified": True,
        "longest_side_input_resize": longest_side or None,
        "no_archive_extraction": "official HF format is TSV; fixed row identity and safe output paths are checked",
        "scope": "CPU source/image/GT verification only; no model inference, GPU training, or image redistribution",
    }
    receipt_path = output / "prepare-receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest_sha256": digest(manifest),
                "images": len(extracted),
                "source_bytes": sum(x["bytes"] for x in downloaded),
            },
            ensure_ascii=False,
        )
    )
    return frozen, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "docs/natural-assistant/external-ocr.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/natural-extension/external-ocr"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/natural-extension/external-ocr-cache"))
    parser.add_argument(
        "--longest-side", type=int, default=0, help="0 keeps original; 600 enables an explicit resized-input comparison"
    )
    args = parser.parse_args()
    if args.longest_side < 0:
        raise ValueError("longest-side must be zero or positive")
    prepare(args.metadata, args.output, args.cache_dir, args.longest_side)


if __name__ == "__main__":
    main()
