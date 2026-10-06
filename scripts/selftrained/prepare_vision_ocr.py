#!/usr/bin/env python3
"""Download licensed pixels/fonts and build finite, source-isolated vision/OCR data.

No pretrained model is used. Run from the checkout, for example:
    python scripts/selftrained/prepare_vision_ocr.py
    python scripts/selftrained/prepare_vision_ocr.py --verify-only

``image_layout`` and ``roi`` are public geometry. ``supervision`` and the last
assistant message are training/scoring targets, never inference prompt inputs.
The image filenames are opaque content hashes and contain no answer strings.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import random
import struct
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pillow_version

REPO = Path(__file__).resolve().parents[2]
FASHION_REVISION = "b2617bb6d3ffa2e429640350f613e3291e10b141"
NOTO_REVISION = "f8d157532fbfaeda587e826d4cd5b21a49186f7c"
VERSION = "phase5-vision-ocr-v1"
SEED = 20261006
CHARACTERS = "大小上下左右開關入出人口"
FASHION_CLASSES = {1: (0, "褲子"), 8: (1, "包"), 9: (2, "短靴")}
FASHION_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
PINNED_SHA256 = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    "fashion-LICENSE": "13ef4788476d292858fa60eb9a5f74aeca5c65770bc885ccaa05823a17ef7be1",
    "fashion-README.md": "bdf17df9bd8b40c09165c7068b8532032094e0015b54ac09233d0ff8397181ea",
    "NotoSansCJKtc-Regular.otf": "dce08bd4fd91aa8aa76ed8fea4b694c2dfb8550f67871e326843212ddbeb88b4",
    "NotoSerifCJKtc-Regular.otf": "234301038e76e7c35c43113785024700c4e4fe7bdce1d1fbbc42fca7e6683798",
    "noto-LICENSE": "6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def opaque(*items: object) -> str:
    return digest(json.dumps(items, ensure_ascii=False, sort_keys=True).encode())[:24]


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(url: str, path: Path, expected_md5: str | None = None, expected_sha256: str | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".download")
        request = urllib.request.Request(url, headers={"User-Agent": "tiny-perceptron-selftrained-data/1"})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        temporary.replace(path)
    raw = path.read_bytes()
    actual_md5 = hashlib.md5(raw).hexdigest()
    if expected_md5 is not None and actual_md5 != expected_md5:
        raise ValueError(f"Authoritative Fashion-MNIST MD5 mismatch: {path}")
    if expected_sha256 is not None and digest(raw) != expected_sha256:
        raise ValueError(f"Pinned source SHA256 mismatch: {path}")
    return {
        "url": url,
        "path": path.as_posix(),
        "bytes": len(raw),
        "sha256": digest(raw),
        "md5": actual_md5,
        "expected_md5": expected_md5,
        "expected_sha256": expected_sha256,
    }


def acquire(data: Path) -> tuple[list[dict], dict[str, Path], list[dict]]:
    fashion_sources = []
    base = f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{FASHION_REVISION}"
    for name, md5 in FASHION_FILES.items():
        source = download(
            f"{base}/data/fashion/{name}", data / "source-cache/fashion-mnist" / name, md5, PINNED_SHA256[name]
        )
        source["path"] = Path(source["path"]).relative_to(data).as_posix()
        fashion_sources.append(source)
    for name in ["LICENSE", "README.md"]:
        source = download(
            f"{base}/{name}", data / "licenses/fashion-mnist" / name, expected_sha256=PINNED_SHA256[f"fashion-{name}"]
        )
        source["path"] = Path(source["path"]).relative_to(data).as_posix()
        fashion_sources.append(source)
    fonts: dict[str, Path] = {}
    font_sources = []
    noto_base = f"https://raw.githubusercontent.com/notofonts/noto-cjk/{NOTO_REVISION}"
    for family in ["Sans", "Serif"]:
        font_name = f"Noto{family}CJKtc-Regular.otf"
        font_path = data / "fonts" / font_name
        fonts[family] = font_path
        for upstream, target in [
            (f"{family}/OTF/TraditionalChinese/{font_name}", font_path),
            (f"{family}/LICENSE", data / "licenses/noto-cjk" / f"{family}-LICENSE.txt"),
        ]:
            source = download(
                f"{noto_base}/{upstream}",
                target,
                expected_sha256=PINNED_SHA256[font_name if upstream.endswith(".otf") else "noto-LICENSE"],
            )
            source["path"] = Path(source["path"]).relative_to(data).as_posix()
            source["family"] = family
            font_sources.append(source)
    return fashion_sources, fonts, font_sources


def load_idx(data: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    directory = data / "source-cache/fashion-mnist"
    image_raw = gzip.decompress((directory / f"{prefix}-images-idx3-ubyte.gz").read_bytes())
    label_raw = gzip.decompress((directory / f"{prefix}-labels-idx1-ubyte.gz").read_bytes())
    magic, count, height, width = struct.unpack(">IIII", image_raw[:16])
    label_magic, label_count = struct.unpack(">II", label_raw[:8])
    if (magic, height, width, label_magic, label_count) != (2051, 28, 28, 2049, count):
        raise ValueError(f"Unexpected Fashion-MNIST IDX headers: {prefix}")
    images = np.frombuffer(image_raw[16:], dtype=np.uint8).reshape(count, height, width)
    labels = np.frombuffer(label_raw[8:], dtype=np.uint8)
    return images, labels


def save_image(data: Path, family: str, image: Image.Image) -> str:
    # Include shape and raw pixels in the opaque name, never source labels/text.
    name = digest(str(image.size).encode() + image.tobytes())[:32]
    relative = Path("images") / family / f"{name}.png"
    path = data / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=False)
    return relative.as_posix()


def record(split: str, task: str, group: str, prompt: str, target: str, image: str, **extra: object) -> dict:
    return {
        "id": opaque(VERSION, split, task, group, prompt, image),
        "group_id": group,
        "split": split,
        "task": task,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}],
        "image": image,
        **extra,
    }


def prepare_vision(data: Path, counts: dict[str, int]) -> tuple[list[dict], dict]:
    official_train, train_labels = load_idx(data, "train")
    official_test, test_labels = load_idx(data, "t10k")
    selected = defaultdict(lambda: defaultdict(list))
    sources = []
    used_pixels: set[str] = set()
    skipped_duplicates = Counter()
    for label, (local_id, _) in FASHION_CLASSES.items():
        for original_split, pixels, labels, destinations in [
            ("train", official_train, train_labels, ["train", "validation"]),
            ("t10k", official_test, test_labels, ["test"]),
        ]:
            indices = np.flatnonzero(labels == label).tolist()
            random.Random(SEED + label + (100 if original_split == "t10k" else 0)).shuffle(indices)
            cursor = 0
            for split in destinations:
                while len(selected[split][local_id]) < counts[split]:
                    if cursor == len(indices):
                        raise ValueError(f"Not enough unique original sources for {split}/{label}")
                    index = indices[cursor]
                    cursor += 1
                    image_hash = digest(pixels[index].tobytes())
                    if image_hash in used_pixels:
                        skipped_duplicates[f"{original_split}:{label}"] += 1
                        continue
                    used_pixels.add(image_hash)
                    source_id = f"fashion-mnist:{original_split}:{index:05d}"
                    selected[split][local_id].append((source_id, pixels[index]))
                    sources.append(
                        {
                            "source_id": source_id,
                            "official_split": original_split,
                            "original_index": index,
                            "split": split,
                            "original_label": label,
                            "local_label": local_id,
                            "pixel_sha256": image_hash,
                        }
                    )
    rows = []
    scenes = []
    for split in ["train", "validation", "test"]:
        for local_id, name in [value for value in FASHION_CLASSES.values()]:
            for source_id, pixels in selected[split][local_id]:
                image = Image.new("L", (112, 112), 0)
                image.paste(Image.fromarray(pixels), (14, 42))
                relative = save_image(data, "vision", image)
                group = opaque("vision-single", source_id)
                rows.append(
                    record(
                        split,
                        "vision_clothing",
                        group,
                        "圖片中的服飾是什麼？",
                        f"這是{name}。",
                        relative,
                        image_layout={"axis": "horizontal", "slots": [[0, 0, 56, 112], [56, 0, 112, 112]]},
                        supervision={
                            "vision_labels": [local_id, -1],
                            "source_ids": [source_id],
                            "query_slot": 0,
                            "semantic_all": [name],
                        },
                    )
                )
                scenes.append({"image": relative, "split": split, "group_id": group, "source_ids": [source_id]})
        count = counts[split]
        if count % 2:
            raise ValueError("Per-class Fashion-MNIST counts must be even for balanced, unique-source pairs")
        half = count // 2
        for first_class in range(3):
            second_class = (first_class + 1) % 3
            for index in range(half):
                first = selected[split][first_class][index]
                second = selected[split][second_class][index + half]
                group = opaque("vision-pair", first[0], second[0])
                for axis, slots, positions in [
                    ("horizontal", [[0, 0, 56, 112], [56, 0, 112, 112]], ["左", "右"]),
                    ("vertical", [[0, 0, 112, 56], [0, 56, 112, 112]], ["上", "下"]),
                ]:
                    for swapped in [False, True]:
                        objects = [(first, first_class), (second, second_class)]
                        if swapped:
                            objects.reverse()
                        image = Image.new("L", (112, 112), 0)
                        for slot, ((_, pixels), _) in zip(slots, objects, strict=True):
                            x0, y0, x1, y1 = slot
                            image.paste(Image.fromarray(pixels), ((x0 + x1 - 28) // 2, (y0 + y1 - 28) // 2))
                        relative = save_image(data, "vision", image)
                        labels = [item[1] for item in objects]
                        source_ids = [item[0][0] for item in objects]
                        for query_slot, position in enumerate(positions):
                            name = list(FASHION_CLASSES.values())[labels[query_slot]][1]
                            rows.append(
                                record(
                                    split,
                                    "vision_relation",
                                    group,
                                    f"{position}邊是什麼？",
                                    f"{position}邊是{name}。",
                                    relative,
                                    image_layout={"axis": axis, "slots": slots},
                                    supervision={
                                        "vision_labels": labels,
                                        "source_ids": source_ids,
                                        "query_slot": query_slot,
                                        "swapped": swapped,
                                        "pair_id": opaque("vision-swap-control", group, axis, query_slot),
                                        "control_type": "swap",
                                        "semantic_all": [name, position],
                                    },
                                )
                            )
                        scenes.append({"image": relative, "split": split, "group_id": group, "source_ids": source_ids})
    return rows, {
        "original_sources": sources,
        "scenes": scenes,
        "selected_original_counts_per_class": counts,
        "skipped_duplicate_original_pixels": dict(skipped_duplicates),
        "class_mapping": [
            {"original_id": original, "local_id": local, "name": name}
            for original, (local, name) in FASHION_CLASSES.items()
        ],
        "split_policy": "Select original IDs and deduplicate original pixel hashes before creating any scene. Validation comes from original train IDs; test only from official t10k IDs. All pair/swap/orientation derivatives stay in the source split.",
        "modifications": "Original 28x28 grayscale uint8 pixels are pasted without resizing into a black 112x112 canvas. Single images use slot 0; slot 1 is blank and its loss label is -1. Paired scenes use fixed horizontal/vertical geometry and both orders, with two public position questions per image.",
        "limits": "Only Fashion-MNIST labels 1/8/9: trousers, bags, ankle boots. Not color images, natural photographs, material recognition, or open object detection.",
    }


def word_pools() -> dict[str, list[str]]:
    selected = {split: [] for split in ["train", "validation", "test"]}
    per_length = {"train": 300, "validation": 40, "test": 60}
    for length in [2, 3, 4]:
        words = ["".join(chars) for chars in itertools.product(CHARACTERS, repeat=length)]
        random.Random(SEED + length).shuffle(words)
        # Length two has 144 combinations, so use 90/24/30 rather than 300/40/60.
        counts = {"train": 90, "validation": 24, "test": 30} if length == 2 else per_length
        offset = 0
        for split, count in counts.items():
            selected[split].extend(words[offset : offset + count])
            offset += count
    for split, words in selected.items():
        random.Random(SEED + len(split)).shuffle(words)
    return selected


def draw_line(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, x: int, y: int, jitter: int
) -> list[list[int]]:
    boxes = []
    for position, char in enumerate(text):
        left, top, right, bottom = font.getbbox(char)
        dx = x + position * 36 + (36 - (right - left)) // 2 - left + jitter
        dy = y + (36 - (bottom - top)) // 2 - top
        draw.text((dx, dy), char, fill=0, font=font)
        box = list(draw.textbbox((dx, dy), char, font=font))
        if not (x + position * 36 <= box[0] < box[2] <= x + (position + 1) * 36 and y <= box[1] < box[3] <= y + 36):
            raise ValueError("Rendered glyph escaped its public cell")
        boxes.append(box)
    return boxes


def prepare_ocr(data: Path, fonts: dict[str, Path], heldout_single_chars: bool) -> tuple[list[dict], dict]:
    rows, renders = [], []
    pools = word_pools()
    variants = {
        "train": [("Sans", 24, 255, 8, 4), ("Sans", 26, 255, 12, 8)],
        "validation": [("Sans", 22, 240, 18, 8), ("Serif", 24, 255, 18, 8)],
        "test": [("Sans", 28, 232, 30, 12), ("Serif", 26, 255, 30, 12)],
    }
    for split in ["train", "validation", "test"]:
        words = pools[split]
        if len(words) % 2:
            raise ValueError("OCR pool must have even size for same-image ROI pairs")
        for index in range(0, len(words), 2):
            texts = words[index : index + 2]
            chosen = [variants[split][(index // 2) % 2]] if split == "train" else variants[split]
            for family, size, background, x, y in chosen:
                font = ImageFont.truetype(str(fonts[family]), size)
                image = Image.new("L", (192, 96), background)
                draw = ImageDraw.Draw(image)
                glyphs = [
                    draw_line(draw, text, font, x, y + row * 42, (index % 3) - 1) for row, text in enumerate(texts)
                ]
                relative = save_image(data, "ocr", image)
                source_id = opaque("ocr-render", split, texts, family, size, background, x, y, index)
                group = opaque("ocr-composition", split, texts)
                for row, text in enumerate(texts):
                    roi = [x, y + row * 42, x + len(text) * 36, y + row * 42 + 36]
                    rows.append(
                        record(
                            split,
                            "ocr",
                            group,
                            f"請讀出區域 {roi} 內的字。",
                            text,
                            relative,
                            roi=roi,
                            supervision={
                                "ocr_text": text,
                                "glyph_boxes": glyphs[row],
                                "render_source_id": source_id,
                                "case": "composition",
                                "pair_id": source_id,
                                "control_type": "roi",
                                "font_family": family,
                                "font_size": size,
                                "background": background,
                                "condition": "unseen_font" if family == "Serif" else "size_background_layout",
                            },
                        )
                    )
                renders.append(
                    {
                        "render_source_id": source_id,
                        "group_id": group,
                        "split": split,
                        "image": relative,
                        "font_family": family,
                        "font_size": size,
                        "background": background,
                        "origin_xy": [x, y],
                        "texts": texts,
                        "case": "composition",
                    }
                )
        if split != "train" and not heldout_single_chars:
            continue
        for char_index, char in enumerate(CHARACTERS):
            repetition_count = 8 if split == "train" else 2
            for repetition in range(repetition_count):
                family, size, background, x, y = variants[split][repetition % 2]
                x += (repetition // 2) * 2
                y += (char_index % 2) * 3
                font = ImageFont.truetype(str(fonts[family]), size)
                image = Image.new("L", (192, 96), background)
                draw = ImageDraw.Draw(image)
                boxes = draw_line(draw, char, font, x, y, (repetition % 3) - 1)
                relative = save_image(data, "ocr", image)
                source_id = opaque("ocr-single-render", split, char_index, repetition)
                group = opaque("ocr-single", split, char_index, repetition)
                roi = [x, y, x + 36, y + 36]
                rows.append(
                    record(
                        split,
                        "ocr",
                        group,
                        f"請讀出區域 {roi} 內的字。",
                        char,
                        relative,
                        roi=roi,
                        supervision={
                            "ocr_text": char,
                            "glyph_boxes": boxes,
                            "render_source_id": source_id,
                            "case": "single_character",
                            "font_family": family,
                            "font_size": size,
                            "background": background,
                            "condition": "unseen_font" if family == "Serif" else "size_background_layout",
                        },
                    )
                )
                renders.append(
                    {
                        "render_source_id": source_id,
                        "group_id": group,
                        "split": split,
                        "image": relative,
                        "font_family": family,
                        "font_size": size,
                        "background": background,
                        "origin_xy": [x, y],
                        "texts": [char],
                        "case": "single_character",
                    }
                )
    return rows, {
        "characters": CHARACTERS,
        "character_to_id": {char: index + 1 for index, char in enumerate(CHARACTERS)},
        "ctc_blank_id": 0,
        "word_pools": pools,
        "renders": renders,
        "render_variants": variants,
        "heldout_single_chars": heldout_single_chars,
        "split_policy": "All 2-4-character target strings are disjoint across train/validation/test. Synthetic render source IDs, complete image pixels and group IDs never cross splits. All derivatives and same-image ROI questions stay grouped. Validation/test use unseen size/background/layout, plus an independently reported held-out Noto Serif family.",
        "single_character_policy": "The 12 known characters are intentionally shared in single-character perception diagnostics; heldout single-character records, when enabled, isolate rendered sources and layout/font, not the known character vocabulary. Composition target strings remain strictly isolated.",
        "modifications": "Unmodified OFL font binaries render author-created strings on 192x96 grayscale canvases. Fixed 36x36 cells; only the user-specified contiguous 1-4-cell ROI is recognized. Two separate rows produce paired questions with different ROIs on the exact same image. Glyph boxes are supervision only; runtime does not use boxes or target length.",
        "limits": "Only 12 printed traditional characters; 2-4-character horizontal composition and explicit ROI. No handwriting, arbitrary characters, photographs, sign detection, vertical text, or unrestricted OCR.",
    }


def write_jsonl(data: Path, family: str, rows: list[dict]) -> list[dict]:
    files = []
    for split in ["train", "validation", "test"]:
        path = data / f"{family}-{split}.jsonl"
        selected = [row for row in rows if row["split"] == split]
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        files.append(
            {
                "path": path.name,
                "records": len(selected),
                "sha256": digest(path.read_bytes()),
                "bytes": path.stat().st_size,
            }
        )
    return files


def summarize(rows: list[dict]) -> dict:
    return {
        "records": len(rows),
        "records_by_split": dict(Counter(row["split"] for row in rows)),
        "records_by_task": dict(Counter(row["task"] for row in rows)),
        "records_by_split_task": dict(Counter(f"{row['split']}:{row['task']}" for row in rows)),
        "paired_controls_by_split": {
            split: len(
                {
                    row["supervision"]["pair_id"]
                    for row in rows
                    if row["split"] == split and "pair_id" in row["supervision"]
                }
            )
            for split in ["train", "validation", "test"]
        },
        "ocr_records_by_split_case_font": dict(
            Counter(
                f"{row['split']}:{row['supervision']['case']}:{row['supervision']['font_family']}"
                for row in rows
                if row["task"] == "ocr"
            )
        ),
        "ocr_records_by_split_length": dict(
            Counter(f"{row['split']}:{len(row['supervision']['ocr_text'])}" for row in rows if row["task"] == "ocr")
        ),
        "unique_images_by_split": {
            split: len({row["image"] for row in rows if row["split"] == split})
            for split in ["train", "validation", "test"]
        },
        "unique_groups_by_split": {
            split: len({row["group_id"] for row in rows if row["split"] == split})
            for split in ["train", "validation", "test"]
        },
    }


def image_inventory(data: Path, rows: list[dict]) -> list[dict]:
    files = []
    for relative in sorted({row["image"] for row in rows}):
        path = data / relative
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path.read_bytes())})
    return files


def verify(data: Path, manifests: Path) -> dict:
    checks: list[str] = []
    all_ids: set[str] = set()
    report = {}
    for family in ["vision", "ocr"]:
        manifest = json.loads((manifests / f"{family}-sources.json").read_text())
        if manifest["generator_sha256"] != digest(Path(__file__).read_bytes()):
            raise ValueError("Generator changed; regenerate the version-bound dataset before verification")
        if family == "ocr" and (
            manifest["ctc_blank_id"] != 0
            or manifest["character_to_id"] != {char: index + 1 for index, char in enumerate(CHARACTERS)}
        ):
            raise ValueError("OCR CTC contract requires blank=0 and known character IDs 1..12")
        rows = []
        for file in manifest["jsonl_files"]:
            path = data / file["path"]
            if digest(path.read_bytes()) != file["sha256"]:
                raise ValueError(f"Dataset JSONL checksum changed: {path}")
            loaded = [json.loads(line) for line in path.read_text().splitlines()]
            if len(loaded) != file["records"]:
                raise ValueError(f"Dataset count changed: {path}")
            rows.extend(loaded)
        for source in manifest["downloads"]:
            path = data / source["path"]
            if digest(path.read_bytes()) != source["sha256"]:
                raise ValueError(f"Downloaded source checksum changed: {path}")
            if source["sha256"] != source["expected_sha256"]:
                raise ValueError(f"Source checksum is not the pinned upstream artifact: {path}")
        if {entry["path"] for entry in manifest["image_files"]} != {row["image"] for row in rows}:
            raise ValueError("Image checksum inventory is incomplete")
        for entry in manifest["image_files"]:
            if digest((data / entry["path"]).read_bytes()) != entry["sha256"]:
                raise ValueError(f"Generated PNG checksum changed: {entry['path']}")
        source_lookup = {}
        if family == "vision":
            for official_split in ["train", "t10k"]:
                pixels, labels = load_idx(data, official_split)
                for source in manifest["original_sources"]:
                    if source["official_split"] == official_split:
                        index = source["original_index"]
                        if (
                            digest(pixels[index].tobytes()) != source["pixel_sha256"]
                            or int(labels[index]) != source["original_label"]
                        ):
                            raise ValueError("Original Fashion pixels or label do not match source manifest")
                        source_lookup[source["source_id"]] = (source, pixels[index])
        groups, images, originals, words, render_ids = [defaultdict(set) for _ in range(5)]
        image_hashes = defaultdict(set)
        for row in rows:
            if row["id"] in all_ids:
                raise ValueError("Duplicate record ID")
            all_ids.add(row["id"])
            if row["split"] not in {"train", "validation", "test"}:
                raise ValueError("Invalid split")
            if row["messages"][-1]["role"] != "assistant" or any(
                item["role"] not in {"user", "assistant", "system", "tool"} for item in row["messages"]
            ):
                raise ValueError("Invalid ordered messages/target")
            relative = Path(row["image"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Image must be a safe path relative to asset root")
            if any(char in relative.name for char in CHARACTERS) or any(
                name in relative.name for _, name in FASHION_CLASSES.values()
            ):
                raise ValueError("Answer leaked in image filename")
            split = row["split"]
            groups[split].add(row["group_id"])
            images[split].add(row["image"])
            with Image.open(data / relative) as image:
                if image.mode != "L" or image.size != ((112, 112) if family == "vision" else (192, 96)):
                    raise ValueError("Unexpected image mode/shape")
                image_hashes[split].add(digest(image.tobytes()))
                if family == "ocr":
                    roi = row["roi"]
                    target = row["supervision"]["ocr_text"]
                    if target != row["messages"][-1]["content"] or not set(target) <= set(CHARACTERS):
                        raise ValueError("OCR target mismatch/unknown character")
                    if (roi[2] - roi[0], roi[3] - roi[1]) != (len(target) * 36, 36):
                        raise ValueError("OCR public cell/ROI contract changed")
                    if not (0 <= roi[0] < roi[2] <= 192 and 0 <= roi[1] < roi[3] <= 96):
                        raise ValueError("OCR ROI outside image")
                    for box in row["supervision"]["glyph_boxes"]:
                        if not (roi[0] <= box[0] < box[2] <= roi[2] and roi[1] <= box[1] < box[3] <= roi[3]):
                            raise ValueError("OCR supervised glyph outside ROI")
                    if len(target) > 1:
                        words[split].add(target)
                    render_ids[split].add(row["supervision"]["render_source_id"])
                else:
                    originals[split].update(row["supervision"]["source_ids"])
                    labels = row["supervision"]["vision_labels"]
                    if len(labels) != 2 or not set(labels) <= {-1, 0, 1, 2}:
                        raise ValueError("Invalid supervised vision category IDs")
                    for slot_index, (slot, label) in enumerate(zip(row["image_layout"]["slots"], labels, strict=True)):
                        if label == -1:
                            if image.crop(slot).getbbox() is not None:
                                raise ValueError("Ignored vision slot is not empty")
                            continue
                        source, original_pixels = source_lookup[row["supervision"]["source_ids"][slot_index]]
                        if source["split"] != split or source["local_label"] != label:
                            raise ValueError("Scene category/source split does not match original")
                        x0, y0, x1, y1 = slot
                        x, y = (x0 + x1 - 28) // 2, (y0 + y1 - 28) // 2
                        if not np.array_equal(np.asarray(image.crop((x, y, x + 28, y + 28))), original_pixels):
                            raise ValueError("Scene failed to preserve original 28x28 pixels")
        for first, second in itertools.combinations(["train", "validation", "test"], 2):
            for kind, sets in [
                ("groups", groups),
                ("image_paths", images),
                ("image_pixels", image_hashes),
                ("original_ids", originals),
                ("composition_strings", words),
                ("render_source_ids", render_ids),
            ]:
                if sets[first] & sets[second]:
                    raise ValueError(f"{family} leakage across {first}/{second}: {kind}")
        by_pair = defaultdict(list)
        for row in rows:
            if "pair_id" in row["supervision"]:
                by_pair[row["supervision"]["pair_id"]].append(row)
        for paired in by_pair.values():
            if len(paired) != 2 or paired[0]["split"] != paired[1]["split"]:
                raise ValueError("Control must contain exactly two records in one split")
            first, second = paired
            if first["messages"][-1]["content"] == second["messages"][-1]["content"]:
                raise ValueError("Control did not change the expected answer")
            if family == "vision" and (
                first["messages"][0] != second["messages"][0] or first["image"] == second["image"]
            ):
                raise ValueError("Swap control must hold the question fixed and change actual pixels")
            if family == "ocr" and (first["image"] != second["image"] or first["roi"] == second["roi"]):
                raise ValueError("ROI control must hold the exact image fixed and change its public ROI")
        if family == "vision":
            pixels_by_split = defaultdict(set)
            for source in manifest["original_sources"]:
                pixels_by_split[source["split"]].add(source["pixel_sha256"])
                if source["split"] == "test" and source["official_split"] != "t10k":
                    raise ValueError("Test contains original training source")
            for first, second in itertools.combinations(["train", "validation", "test"], 2):
                if pixels_by_split[first] & pixels_by_split[second]:
                    raise ValueError("Original source pixel duplication across splits")
            checks.append("Fashion original IDs and original pixel hashes isolated before scene assembly")
        else:
            by_image = defaultdict(list)
            for row in rows:
                by_image[row["image"]].append(row)
            for split in ["train", "validation", "test"]:
                paired = [items for items in by_image.values() if len(items) == 2 and items[0]["split"] == split]
                if not paired or not any(
                    items[0]["roi"] != items[1]["roi"]
                    and items[0]["messages"][-1]["content"] != items[1]["messages"][-1]["content"]
                    for items in paired
                ):
                    raise ValueError("Missing same-image/different-ROI contrast")
            checks.append("OCR composition strings isolated; same-image different-ROI contrasts present in every split")
        report[family] = summarize(rows)
    return {
        "status": "passed",
        "checks": checks
        + [
            "all source/JSONL/generated-PNG checksums",
            "record IDs/messages",
            "safe opaque filenames",
            "actual image modes/shapes",
            "source/group/image/word isolation",
            "ROI bounds and supervised glyph alignment",
            "original 28x28 scene pixels and class labels",
            "exactly-two-record swap/ROI controls with changed answers",
            "CTC blank=0 and character IDs 1..12",
        ],
        "data": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "outputs/selftrained/data")
    parser.add_argument("--manifest-dir", type=Path, default=REPO / "outputs/selftrained/manifests")
    parser.add_argument("--train-per-class", type=int, default=400)
    parser.add_argument("--validation-per-class", type=int, default=80)
    parser.add_argument("--test-per-class", type=int, default=120)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--omit-heldout-single-characters",
        action="store_true",
        help="Strictly isolate all target strings; test/validation then contain compositions only",
    )
    args = parser.parse_args()
    data, manifests = args.data_dir.resolve(), args.manifest_dir.resolve()
    if not args.verify_only:
        data.mkdir(parents=True, exist_ok=True)
        fashion_sources, fonts, font_sources = acquire(data)
        vision_rows, vision = prepare_vision(
            data, {"train": args.train_per_class, "validation": args.validation_per_class, "test": args.test_per_class}
        )
        ocr_rows, ocr = prepare_ocr(data, fonts, not args.omit_heldout_single_characters)
        common = {
            "version": VERSION,
            "seed": SEED,
            "generator": "scripts/selftrained/prepare_vision_ocr.py",
            "generator_sha256": digest(Path(__file__).read_bytes()),
            "asset_root": data.relative_to(REPO).as_posix() if data.is_relative_to(REPO) else data.as_posix(),
            "software": {"numpy": np.__version__, "Pillow": pillow_version},
            "inference_contract": "Only messages before the final assistant target, actual image pixels, image_layout and/or roi enter inference. ID/path names only locate assets. Never pass supervision or manifest labels/text into the model prompt. No pretrained weights or OCR/CLIP/ASR model are used in data preparation.",
        }
        dump_json(
            manifests / "vision-sources.json",
            {
                **common,
                "source_revision": FASHION_REVISION,
                "license": "MIT",
                "attribution": "Fashion-MNIST, Han Xiao, Kashif Rasul, Roland Vollgraf, Zalando Research (2017); original MIT copyright/license retained.",
                "downloads": fashion_sources,
                **vision,
                "jsonl_files": write_jsonl(data, "vision", vision_rows),
                "image_files": image_inventory(data, vision_rows),
                "counts": summarize(vision_rows),
            },
        )
        dump_json(
            manifests / "ocr-sources.json",
            {
                **common,
                "source_revision": NOTO_REVISION,
                "license": "SIL Open Font License 1.1",
                "attribution": "Noto CJK Sans/Serif Traditional Chinese regular fonts; copyright notices and both unmodified OFL license files retained. Rendered strings were authored by this deterministic generator.",
                "downloads": font_sources,
                **ocr,
                "jsonl_files": write_jsonl(data, "ocr", ocr_rows),
                "image_files": image_inventory(data, ocr_rows),
                "counts": summarize(ocr_rows),
            },
        )
    result = verify(data, manifests)
    dump_json(manifests / "vision-ocr-integrity.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
