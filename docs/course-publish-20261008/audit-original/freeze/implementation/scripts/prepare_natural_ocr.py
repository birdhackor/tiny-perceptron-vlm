"""準備課程自行寫作的繁體中文 OCR 教具；合成圖片不代表自然招牌 OCR。"""

import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_REVISION = "f8d157532fbfaeda587e826d4cd5b21a49186f7c"
FONT_ROOT = f"https://raw.githubusercontent.com/notofonts/noto-cjk/{FONT_REVISION}"
FONT_LICENSE_SHA256 = "6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2"
FONT_FILES = {
    "sans": (
        "Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf",
        "dce08bd4fd91aa8aa76ed8fea4b694c2dfb8550f67871e326843212ddbeb88b4",
    ),
    "serif": (
        "Serif/OTF/TraditionalChinese/NotoSerifCJKtc-Regular.otf",
        "234301038e76e7c35c43113785024700c4e4fe7bdce1d1fbbc42fca7e6683798",
    ),
}

TRAIN_TEXT = [
    "臺灣學生一起讀書。",
    "今天圖書館正常開放。",
    "請把課本放在桌上。",
    "老師正在黑板前上課。",
    "這裡可以借書與還書。",
    "教室有三張桌子。",
    "我們下課後去操場。",
    "這本書介紹圖片與聲音。",
    "公車站就在學校旁邊。",
    "請依箭頭方向走。",
    "下午兩點開始上課。",
    "餐廳今天提供熱湯。",
    "請先洗手再吃飯。",
    "天氣晴朗適合散步。",
    "紅色杯子放在左邊。",
    "藍色書包放在右邊。",
    "有人正在閱讀報紙。",
    "這張圖片沒有文字。",
    "安全出口請保持暢通。",
    "會議室裡有四把椅子。",
]
VALIDATION_TEXT = [
    "請在圖書館裡保持安靜。",
    "藍色杯子就在桌子右邊。",
    "下午請一起去操場散步。",
    "今天可以借書與還書。",
    "學生正在教室裡讀書。",
    "請先閱讀這本課本。",
]
TEST_TEXT = [
    "圖書館下午兩點開放。",
    "請把紅色課本放在左邊。",
    "有人正在公車站讀報紙。",
    "今天餐廳有熱湯。",
    "臺灣學生在操場散步。",
    "藍色杯子放在桌子上。",
    "會議室今天正常開放。",
    "老師正在閱讀這本書。",
    "請先洗手再借書。",
    "學校旁邊有安全出口。",
    "聲音與圖片都可以學。",
    "這張圖片有中文字。",
]
TRAIN_PAGES = [
    ["請先借書", "再到教室讀書"],
    ["圖書館開放", "下午兩點上課"],
    ["左邊是紅色杯子", "右邊是藍色書包"],
    ["先洗手", "再吃飯"],
    ["今天有熱湯", "餐廳正常開放"],
    ["老師在黑板前", "學生在桌子旁"],
    ["公車站在左邊", "學校在右邊"],
    ["請保持安靜", "有人正在讀書"],
]
VALIDATION_PAGES = [
    ["先到操場", "再回教室"],
    ["杯子在右邊", "課本在左邊"],
    ["今天先讀書", "下午再上課"],
]
TEST_PAGES = [
    ["先到圖書館", "再去公車站"],
    ["紅色書包在右邊", "藍色杯子在左邊"],
    ["先借這本書", "再讀這張報紙"],
    ["下午先散步", "然後回教室"],
    ["今天餐廳開放", "明天一起上課"],
    ["先閱讀圖片", "再介紹聲音"],
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def download_exact(url, path, expected=None):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    data = path.read_bytes()
    digest = sha256(data)
    if expected and digest != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return {"url": url, "file": str(path), "bytes": len(data), "sha256": digest}


def make_background(size, seed, style):
    rng = random.Random(seed)
    base = [(248, 246, 237), (220, 231, 239), (241, 226, 211), (228, 238, 224)][style % 4]
    image = Image.new("RGB", size)
    image.putdata(
        [tuple(max(0, min(255, channel + rng.randint(-5, 5))) for channel in base) for _ in range(size[0] * size[1])]
    )
    draw = ImageDraw.Draw(image)
    if style % 4 == 1:
        for y in range(16, size[1], 40):
            draw.line((0, y, size[0], y), fill=(170, 180, 193), width=2)
    elif style % 4 == 2:
        for x in range(0, size[0], 60):
            draw.line((x, 0, x, size[1]), fill=(198, 175, 151), width=2)
    elif style % 4 == 3:
        for _ in range(6):
            x = rng.randrange(size[0] - 30)
            y = rng.randrange(size[1] - 30)
            draw.ellipse((x, y, x + 25, y + 25), outline=(160, 181, 159), width=2)
    return image


def render(lines, font_file, seed, style, vertical=False):
    size = (512, 352) if vertical or len(lines) > 1 else (512, 160)
    image = make_background(size, seed, style)
    draw = ImageDraw.Draw(image)
    font_size = 30 if vertical or len(lines) > 1 else 32
    font = ImageFont.truetype(str(font_file), font_size)
    missing = (tuple(font.getmask("\U0010ffff").size), bytes(font.getmask("\U0010ffff")))
    for char in "".join(lines):
        mask = font.getmask(char)
        if (tuple(mask.size), bytes(mask)) == missing:
            raise ValueError(f"font lacks {char!r}")
    boxes = []
    for order, text in enumerate(lines):
        if vertical:
            x = 350 - order * 100
            chars = []
            for index, char in enumerate(text):
                y = 25 + index * (font_size + 3)
                draw.text((x, y), char, font=font, fill=(24, 30, 35))
                chars.append(list(draw.textbbox((x, y), char, font=font)))
            box = [
                min(b[0] for b in chars),
                min(b[1] for b in chars),
                max(b[2] for b in chars),
                max(b[3] for b in chars),
            ]
        else:
            x, y = 30, 40 + order * 85
            box = list(draw.textbbox((x, y), text, font=font))
            if box[2] > size[0] - 15:
                raise ValueError(f"text exceeds image: {text}")
            draw.text((x, y), text, font=font, fill=(24, 30, 35))
        if box[3] > size[1] - 10:
            raise ValueError(f"text exceeds image: {text}")
        boxes.append({"order": order, "text": text, "bbox_xyxy": box})
    return image, boxes


def prepare(output, font_cache, seed=42):
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "licenses").mkdir(exist_ok=True)
    course_license = (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes()
    (output / "licenses" / "course-MIT.txt").write_bytes(course_license)
    fonts = {}
    sources = []
    for name, (relative, expected) in FONT_FILES.items():
        if not expected:
            raise ValueError("official font SHA-256 must be pinned")
        path = font_cache / Path(relative).name
        receipt = download_exact(f"{FONT_ROOT}/{relative}", path, expected)
        receipt["file"] = path.name
        receipt["stored_in"] = "font-cache directory; font binary is not copied into the image package"
        receipt.update({"id": f"noto-{name}-tc", "revision": FONT_REVISION, "license": "OFL-1.1"})
        license_path = output / "licenses" / f"Noto-{name}-OFL.txt"
        receipt["license_file"] = str(license_path.relative_to(output))
        receipt["license_fetch"] = download_exact(
            f"{FONT_ROOT}/{relative.split('/')[0]}/LICENSE", license_path, FONT_LICENSE_SHA256
        )
        receipt["license_fetch"]["file"] = str(license_path.relative_to(output))
        sources.append(receipt)
        fonts[name] = path
    sources.append(
        {
            "id": "course-written-text",
            "license": "MIT",
            "scope": "all phrases, layout annotations and procedural backgrounds written for this course",
            "license_file": "licenses/course-MIT.txt",
            "license_sha256": sha256(course_license),
        }
    )
    rows = []
    image_records = []

    def add(split, family, task, lines, variant=0, vertical=False, image_info=None):
        font_name = "serif" if split == "test" else "sans"
        ident = f"{split}-{task}-{len(rows):04d}"
        if image_info is None:
            style = (variant + len(rows)) % 4
            image, boxes = render(lines, fonts[font_name], seed + len(rows), style, vertical)
            path = output / "images" / f"{ident}.png"
            image.save(path)
            image_info = {
                "image": str(path.relative_to(output)),
                "sha256": sha256(path.read_bytes()),
                "bytes": path.stat().st_size,
                "size": list(image.size),
                "font_source": f"noto-{font_name}-tc" if lines else None,
                "layout": "vertical-right-to-left" if vertical else "horizontal-top-to-bottom",
                "style": style,
                "boxes": boxes,
            }
            image_records.append(image_info)
        if task == "text_presence":
            answer = "有" if image_info["boxes"] else "沒有"
            user = "圖片裡有沒有中文字？有就只回答「有」，沒有就只回答「沒有」。"
            references = {"kind": "exact", "accepted": [answer], "has_chinese_text": bool(image_info["boxes"])}
        else:
            answer = "\n".join(lines)
            references = {
                "kind": "ocr_order" if len(lines) > 1 else "ocr",
                "text": answer,
                "strip_whitespace": len(lines) == 1,
            }
            user = "只輸出圖片中的中文字與標點，不要解釋。"
            if len(lines) > 1:
                user = (
                    "請依直排從右欄到左欄、每欄從上到下的順序抄寫圖片文字；每欄換一行，只輸出文字。"
                    if vertical
                    else "請由上到下抄寫圖片中的兩行文字，保留換行；只輸出文字與標點，不要解釋。"
                )
        row = {
            "id": ident,
            "split": split,
            "task": task,
            "family": family,
            "user": user,
            "answer": answer,
            "image": image_info["image"],
            "references": references,
        }
        rows.append(row)
        return row, image_info

    positives = {}
    for split, phrases, variants in [
        ("train", TRAIN_TEXT, 4),
        ("validation", VALIDATION_TEXT, 1),
        ("test", TEST_TEXT, 1),
    ]:
        positives[split] = []
        for index, phrase in enumerate(phrases):
            family = f"phrase-{sha256(phrase.encode())[:16]}"
            for variant in range(variants):
                row, image_info = add(split, family, "ocr", [phrase], variant)
                positives[split].append((row, image_info))
    for split, yes_count, no_count in [("train", 12, 12), ("validation", 2, 1), ("test", 3, 3)]:
        for row, image_info in positives[split][:yes_count]:
            add(split, row["family"], "text_presence", [], image_info=image_info)
        for index in range(no_count):
            add(split, f"blank-{split}-{index}", "text_presence", [], index)
    for split, pages, variants in [
        ("train", TRAIN_PAGES, 2),
        ("validation", VALIDATION_PAGES, 1),
        ("test", TEST_PAGES, 1),
    ]:
        for index, lines in enumerate(pages):
            family = f"page-{sha256(chr(10).join(lines).encode())[:16]}"
            for variant in range(variants):
                add(split, family, "ocr", lines, variant, vertical=bool(index % 2))
    split_counts = Counter(row["split"] for row in rows)
    if dict(split_counts) != {"train": 120, "validation": 12, "test": 24}:
        raise AssertionError(split_counts)
    for key in ["family", "image"]:
        assignment = {}
        for row in rows:
            if row[key] in assignment and assignment[row[key]] != row["split"]:
                raise AssertionError(f"{key} leaked across splits")
            assignment[row[key]] = row["split"]
    train_chars = set("".join(row["answer"] for row in rows if row["split"] == "train" and row["task"] == "ocr"))
    test_chars = set("".join(row["answer"] for row in rows if row["split"] == "test" and row["task"] == "ocr"))
    manifest = {
        "schema_version": 1,
        "dataset_version": "course-traditional-ocr-v1",
        "seed": seed,
        "license": "MIT for course-written text, generated images and annotations; font files remain OFL-1.1",
        "scope": "synthetic Traditional Chinese short lines, two-line pages, vertical columns and text-free procedural backgrounds; not natural photographs or general Chinese OCR",
        "split_policy": "whole phrase/page family and all derived renders stay on one side; test text uses held-out Noto Serif TC, train/validation use Noto Sans TC",
        "test_caveat": "unseen font/text combinations are not a guarantee of natural scene OCR; no claim that pretrained base models have never seen similar words/fonts",
        "counts": dict(split_counts),
        "counts_by_task_and_reference": dict(
            Counter(f"{row['split']}:{row['task']}:{row['references']['kind']}" for row in rows)
        ),
        "train_characters": "".join(sorted(train_chars)),
        "test_characters_absent_from_train": "".join(sorted(test_chars - train_chars)),
        "sources": sources,
        "images": image_records,
        "rows": rows,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(path),
                "sha256": sha256(path.read_bytes()),
                "counts": dict(split_counts),
                "unique_images": len(image_records),
                "unseen_test_characters": manifest["test_characters_absent_from_train"],
            },
            ensure_ascii=False,
        )
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/natural/ocr"))
    parser.add_argument("--font-cache", type=Path, default=Path("outputs/natural-extension/font-cache"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.output, args.font_cache, args.seed)


if __name__ == "__main__":
    main()
