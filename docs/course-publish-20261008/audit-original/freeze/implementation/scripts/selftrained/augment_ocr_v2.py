#!/usr/bin/env python3
"""Deterministic train-only OCR style augmentation for the isolated V2 candidate.

Copies the complete V1 data directory without altering heldout bytes, retains
every original train row, and adds five renders per original train image. Both
Noto Sans and Serif are trained families in V2: historical heldout condition
fields remain untouched and must not be read as an unseen-font V2 claim.
No downloads, pretrained weights, new strings, or evaluation generation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
VERSION = "phase5-ocr-train-styles-letterbox-v2"
PREPROCESS_VERSION = "gray-crops32-ocr-letterbox-full-logmel40-v2"
SEED = 20261006
VARIANTS = 5
TRAIN_APPEND_PATHS = {"ocr-train.jsonl", "text-tools-train.jsonl", "voice-train.jsonl"}
NOTO_REVISION = "f8d157532fbfaeda587e826d4cd5b21a49186f7c"
FONT_SHA256 = {
    "Sans": "dce08bd4fd91aa8aa76ed8fea4b694c2dfb8550f67871e326843212ddbeb88b4",
    "Serif": "234301038e76e7c35c43113785024700c4e4fe7bdce1d1fbbc42fca7e6683798",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def opaque(*values: object) -> str:
    return sha(json.dumps(values, ensure_ascii=False, sort_keys=True).encode())[:24]


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def style(parent_image: str, index: int) -> dict:
    # Each source has both families. Sizes/photometry are varied independently
    # of its target string, using only the existing opaque image ID and seed.
    rng = random.Random(int(opaque(VERSION, SEED, parent_image, index), 16))
    fixed = [
        ("Sans", 22, 240, 0, 0),
        ("Serif", 24, 255, 16, 0),
        ("Sans", 28, 232, 24, 0),
        ("Serif", 26, 240, 0, 1),
    ]
    family, size, background, ink, stroke = (
        fixed[index]
        if index < 4
        else (
            rng.choice(("Sans", "Serif")),
            rng.choice((20, 22, 24, 26, 28)),
            rng.choice((224, 232, 240, 248, 255)),
            rng.choice((0, 16, 32, 48)),
            rng.choice((0, 1)),
        )
    )
    return {
        "font_family": family,
        "font_size": size,
        "background": background,
        "ink": ink,
        "stroke_width": stroke,
        "jitter_xy": [rng.randint(-2, 2), rng.randint(-2, 2)],
    }


@lru_cache(maxsize=20)
def font(path: str, size: int):
    return ImageFont.truetype(path, size)


def draw_roi(draw: ImageDraw.ImageDraw, text: str, roi: list[int], face, transform: dict) -> list[list[int]]:
    x, y, x1, y1 = roi
    if x1 - x != 36 * len(text) or y1 - y != 36:
        raise ValueError("Original train ROI is not the declared 36px finite-cell geometry")
    jx, jy = transform["jitter_xy"]
    stroke = transform["stroke_width"]
    boxes = []
    for index, char in enumerate(text):
        left, top, right, bottom = face.getbbox(char, stroke_width=stroke)
        dx = x + index * 36 + (36 - (right - left)) // 2 - left + jx
        dy = y + (36 - (bottom - top)) // 2 - top + jy
        draw.text((dx, dy), char, font=face, fill=transform["ink"], stroke_width=stroke, stroke_fill=transform["ink"])
        box = list(draw.textbbox((dx, dy), char, font=face, stroke_width=stroke))
        if not (x + index * 36 <= box[0] < box[2] <= x + (index + 1) * 36 and y <= box[1] < box[3] <= y1):
            raise ValueError("Augmented glyph escaped its original public cell")
        boxes.append(box)
    return boxes


def frozen_parent_files(source: Path, output: Path) -> dict[str, str]:
    """Serialize the frozen package identities, never incidental local caches."""
    frozen = json.loads((ROOT / "docs/selftrained/manifest.json").read_text(encoding="utf-8"))
    recipe = json.loads((ROOT / "docs/selftrained/package-recipe.json").read_text(encoding="utf-8"))
    entries = frozen["records"] + frozen["assets"]
    expected = {entry["path"]: entry for entry in entries}
    packaged = {entry["path"]: entry for entry in recipe["source_files"]}
    if len(expected) != len(entries) or set(expected) != set(packaged):
        raise ValueError("Frozen parent manifest and package recipe list different files")
    identities = {}
    for relative, entry in sorted(expected.items()):
        path = source / relative
        # Notices not emitted into the producers' data root are exact files
        # from the frozen recipe, not newly invented license/notice content.
        if not path.exists():
            path = ROOT / packaged[relative]["source"]
        if (
            path.is_symlink()
            or not path.is_file()
            or entry["sha256"] != packaged[relative]["sha256"]
            or entry["bytes"] != packaged[relative]["bytes"]
            or file_sha(path) != entry["sha256"]
            or path.stat().st_size != entry["bytes"]
        ):
            raise ValueError(f"Source file differs from frozen parent package: {relative}")
        target = output / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        if target.is_symlink():
            raise ValueError(f"Packaged destination must not be a symlink: {relative}")
        if relative in TRAIN_APPEND_PATHS:
            if not target.read_bytes().startswith(path.read_bytes()):
                raise ValueError(f"Frozen train byte prefix changed: {relative}")
        elif file_sha(target) != entry["sha256"]:
            raise ValueError(f"Immutable packaged file changed: {relative}")
        identities[relative] = entry["sha256"]
    return identities


def augment(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and destination must be separate, non-nested directories")
    train_path = source / "ocr-train.jsonl"
    original_raw = train_path.read_bytes()
    original = [json.loads(line) for line in original_raw.decode().splitlines() if line.strip()]
    if not original or any(row["split"] != "train" or row["task"] != "ocr" for row in original):
        raise ValueError("Only original OCR train rows may be augmented")
    source_files = {p.relative_to(source).as_posix(): file_sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    if any((source / name).is_symlink() for name in source_files):
        raise ValueError("Source data must use regular files, not symlinks")
    output.mkdir(parents=True, exist_ok=True)
    for relative, expected in source_files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            # Other reviewed augmenters may already have appended their train
            # rows. Preserve their suffix; only this script owns OCR's suffix.
            if relative in TRAIN_APPEND_PATHS and target.read_bytes().startswith((source / relative).read_bytes()):
                continue
            if file_sha(target) != expected:
                raise ValueError(f"Existing destination differs from original data: {relative}")
        else:
            shutil.copyfile(source / relative, target)
    stable_parent_sha = frozen_parent_files(source, output)
    for family, expected in FONT_SHA256.items():
        path = output / "fonts" / f"Noto{family}CJKtc-Regular.otf"
        if file_sha(path) != expected:
            raise ValueError(f"Pinned OFL font SHA mismatch: {family}")
    by_image = defaultdict(list)
    for row in original:
        by_image[row["image"]].append(row)
    new_rows, renders = [], []
    ids = {row["id"] for row in original}
    for parent_image, parents in sorted(by_image.items()):
        if len({row["group_id"] for row in parents}) != 1:
            raise ValueError("A shared original image must have one train group")
        with Image.open(source / parent_image) as original_image:
            canvas_size = original_image.size
        for index in range(VARIANTS):
            transform = style(parent_image, index)
            image = Image.new("L", canvas_size, transform["background"])
            face = font(
                str(output / "fonts" / f"Noto{transform['font_family']}CJKtc-Regular.otf"), transform["font_size"]
            )
            draw = ImageDraw.Draw(image)
            all_boxes = [
                draw_roi(draw, parent["supervision"]["ocr_text"], parent["roi"], face, transform) for parent in parents
            ]
            pixel_sha = sha(str(image.size).encode() + image.tobytes())
            relative = f"images/ocr-v2-train/{pixel_sha[:32]}.png"
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, optimize=False)
            render_id = opaque(VERSION, parent_image, index, transform, pixel_sha)
            children = []
            for parent, boxes in zip(parents, all_boxes, strict=True):
                row = copy.deepcopy(parent)
                row["id"] = opaque(VERSION, parent["id"], index, render_id)
                if row["id"] in ids:
                    raise ValueError("Augmented ID collision")
                ids.add(row["id"])
                row["image"] = relative
                supervision = row["supervision"]
                supervision.update(
                    {
                        "glyph_boxes": boxes,
                        "render_source_id": render_id,
                        "font_family": transform["font_family"],
                        "font_size": transform["font_size"],
                        "background": transform["background"],
                        "condition": "train_style_augmentation_v2",
                        "ink": transform["ink"],
                        "stroke_width": transform["stroke_width"],
                    }
                )
                if "pair_id" in supervision:
                    supervision["pair_id"] = render_id
                new_rows.append(row)
                children.append({"id": row["id"], "parent_id": parent["id"], "roi": row["roi"]})
            renders.append(
                {
                    "image": relative,
                    "sha256": file_sha(path),
                    "pixel_sha256": pixel_sha,
                    "parent_image": parent_image,
                    "parent_image_sha256": source_files[parent_image],
                    "group_id": parents[0]["group_id"],
                    "render_source_id": render_id,
                    "variant_index": index,
                    "transform": transform,
                    "records": children,
                }
            )
    suffix = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode() for row in new_rows)
    if not original_raw.endswith(b"\n"):
        original_raw += b"\n"
    temporary = output / "ocr-train.jsonl.tmp"
    temporary.write_bytes(original_raw + suffix)
    temporary.replace(output / "ocr-train.jsonl")
    # Other modalities' coordinated train appends remain untouched. Every
    # inherited non-train file must still match its original bytes exactly.
    for relative, expected in source_files.items():
        if relative in TRAIN_APPEND_PATHS:
            if not (output / relative).read_bytes().startswith((source / relative).read_bytes()):
                raise ValueError(f"Original train byte prefix changed: {relative}")
        elif file_sha(output / relative) != expected:
            raise ValueError(f"Original file changed: {relative}")
    if any(file_sha(source / relative) != expected for relative, expected in source_files.items()):
        raise ValueError("Source data changed while augmenting")
    original_groups = {row["group_id"] for row in original}
    original_strings = {row["supervision"]["ocr_text"] for row in original}
    assert len(new_rows) == len(original) * VARIANTS
    assert all(
        row["group_id"] in original_groups
        and row["supervision"]["ocr_text"] in original_strings
        and row["messages"][-1]["content"] == row["supervision"]["ocr_text"]
        for row in new_rows
    )
    manifest = {
        "schema": VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "seed": SEED,
        "variants_per_original_image": VARIANTS,
        "source_file_sha256": stable_parent_sha,
        "parent_manifest": {
            "path": "docs/selftrained/manifest.json",
            "sha256": file_sha(ROOT / "docs/selftrained/manifest.json"),
        },
        "parent_recipe": {
            "path": "docs/selftrained/package-recipe.json",
            "sha256": file_sha(ROOT / "docs/selftrained/package-recipe.json"),
        },
        "output_ocr_train_sha256": file_sha(output / "ocr-train.jsonl"),
        "original_train_rows": len(original),
        "added_train_rows": len(new_rows),
        "total_train_rows": len(original) + len(new_rows),
        "original_train_groups": len(original_groups),
        "additional_target_strings": 0,
        "original_bytes_preserved_except_train_append": True,
        "coordinated_train_append_paths": sorted(TRAIN_APPEND_PATHS),
        "all_derivatives_keep_original_train_group": True,
        "font_sources": [
            {
                "family": family,
                "path": f"fonts/Noto{family}CJKtc-Regular.otf",
                "sha256": expected,
                "url": f"https://raw.githubusercontent.com/notofonts/noto-cjk/{NOTO_REVISION}/{family}/OTF/TraditionalChinese/Noto{family}CJKtc-Regular.otf",
                "license_path": f"licenses/noto-cjk/{family}-LICENSE.txt",
                "license": "OFL-1.1",
            }
            for family, expected in FONT_SHA256.items()
        ],
        "claim_boundary": {
            "trained_font_families": ["Sans", "Serif"],
            "unseen_font_generalization_claim": False,
            "heldout_condition_fields": "Byte-identical historical V1 metadata; unseen_font is not a V2 claim.",
            "composition_strings": "Only original train strings; original heldout targets and bytes remain unchanged.",
            "runtime_inputs": "Pixels and public ROI only; glyph boxes and augmentation parameters are supervision/audit only.",
        },
        "styles_added": dict(sorted(Counter(row["supervision"]["font_family"] for row in new_rows).items())),
        "script_sha256": file_sha(Path(__file__)),
        "renders": renders,
    }
    dump(output.parent / "manifests" / "ocr-v2-augmentation.json", manifest)
    return {
        key: manifest[key]
        for key in (
            "schema",
            "preprocess_version",
            "original_train_rows",
            "added_train_rows",
            "total_train_rows",
            "output_ocr_train_sha256",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=ROOT / "outputs/selftrained/data")
    parser.add_argument("--output-data", type=Path, default=ROOT / "outputs/selftrained-v2/data")
    args = parser.parse_args()
    print(json.dumps(augment(args.source_data, args.output_data), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
