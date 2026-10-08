"""手畫 5×7 數字：每張 16×16 圖片有一到兩位數；同一答案的變體留在同側。"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image

GLYPHS = [
    ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
]


def draw_digits(text, offset=0):
    if not text.isascii() or not text.isdigit() or len(text) > 2:
        raise ValueError("16×16 教具只接受一到兩位數")
    image = Image.new("RGB", (16, 16), "black")
    width = len(text) * 6 - 1
    start = (16 - width) // 2 + offset
    for digit_index, digit in enumerate(text):
        for y, row in enumerate(GLYPHS[int(digit)]):
            for x, bit in enumerate(row):
                if bit == "1":
                    image.putpixel((start + digit_index * 6 + x, 4 + y), (255, 255, 255))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/generated/ocr"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    (args.output / "images").mkdir(parents=True, exist_ok=True)
    families = list(range(100))
    random.Random(args.seed).shuffle(families)
    stats = {}
    for split, selected in (("train", families[:80]), ("validation", families[80:90]), ("test", families[90:])):
        records = []
        for value in selected:
            for offset in (-1, 0, 1):
                path = f"images/{value}-{offset}.png"
                draw_digits(str(value), offset).save(args.output / path)
                records.append(
                    {
                        "image": path,
                        "question": "read digits",
                        "answer": str(value),
                        "family": str(value),
                        "split": split,
                        "source": "course-generated",
                        "license": "MIT",
                    }
                )
        path = args.output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        stats[split] = {
            "records": len(records),
            "families": len(selected),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "seed": args.seed,
        "split_unit": "digit-string family",
        "splits": stats,
        "scope": "fixed 5×7 font, no claim about handwriting or documents",
        "license": "MIT",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
