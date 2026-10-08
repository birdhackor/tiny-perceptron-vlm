"""Deterministic LFS snapshots and a merged manifest for the practical project."""

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "natural-assistant-v3"
KINDS = ("vision", "ocr", "speech")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def package(directory, destination, prefix):
    files = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in sorted(directory.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Data snapshots must not contain symlinks")
                if not path.is_file():
                    continue
                name = f"{prefix}/{path.relative_to(directory).as_posix()}"
                content = path.read_bytes()
                item = tarfile.TarInfo(name)
                item.size, item.mode, item.mtime = len(content), 0o644, 0
                archive.addfile(item, io.BytesIO(content))
                files.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    return files


def text_rows():
    """Self-written conversation replay; every derived turn stays in its family."""
    rows = []
    examples = {
        "train": [
            ("我今天有點累。", "可以先休息一下；想聊聊今天發生了什麼嗎？"),
            ("讀書時一直分心怎麼辦？", "先挑一個小段落讀五分鐘，再短暫休息。"),
            ("請簡單解釋模型在做什麼。", "它根據輸入預測接下來的文字；回答仍可能出錯。"),
            ("晚餐想省事，不吃辣。", "可以選清淡湯麵或煮蛋配青菜。"),
            ("有一個字看不清怎麼辦？", "可以放大或重拍；仍看不清時，應說明是哪個字不確定。"),
            ("你會朗讀回答嗎？", "這個版本輸出文字，沒有接語音合成。"),
            ("我看了一張有人騎腳踏車的照片。", "可以留意他是在騎車、停車，還是做特技，別只數物件。"),
            ("今天下雨，要帶什麼？", "可以帶雨傘或雨衣，也注意路面濕滑。"),
            ("我想先去圖書館，再去公車站。", "順序是先圖書館，再公車站。"),
            ("不要猜圖片裡看不清的文字。", "好，能看清的才抄寫；看不清的部分會說明。"),
        ],
        "validation": [
            ("讀了很久，想休息一下。", "可以休息幾分鐘，喝點水，再決定是否繼續。"),
            ("請用一句話說明語音入口。", "先把錄音辨識成文字，再交給同一個聊天模型。"),
            ("這張圖能證明之前發生什麼嗎？", "單張照片通常只能支持畫面可見的線索，不能確定之前的故事。"),
        ],
        "test": [
            ("我第一次學這個，有點緊張。", "可以先從一個小例子開始，不用一次學完。"),
            ("請給我不辣又簡單的早餐建議。", "可以選吐司、雞蛋和牛奶。"),
            ("如果語音認錯字，我該怎麼辦？", "先檢查逐字稿並更正，避免錯字影響後面的回答。"),
            ("請用繁體中文和我聊天。", "好，我會用繁體中文回答。"),
        ],
    }
    for split, pairs in examples.items():
        for index, (question, answer) in enumerate(pairs):
            family = f"course-chat-{split}-{index:02}"
            for style in (0, 1, 2) if split == "train" else (0,):
                rows.append(
                    {
                        "id": f"{family}-{style}",
                        "family": family,
                        "split": split,
                        "task": "text_chat",
                        "user": question,
                        "answer": answer,
                        "image": None,
                        "history": [{"role": "system", "content": "請用繁體中文簡短回應。"}],
                        "references": {"kind": "manual", "text": answer},
                        "source": "course-self-written-chat-MIT",
                    }
                )
    # Only the final answer is supervised; earlier turns are real context.
    dialogs = {
        "train": [
            ("我不吃辣。", "給我晚餐建議。", "可以吃清淡湯麵或蒸蛋。"),
            ("請叫我小林。", "你記得怎麼稱呼我嗎？", "你希望我叫你小林。"),
        ],
        "validation": [("我不吃肉。", "給我簡單的晚餐建議。", "可以吃蔬菜湯麵或豆腐配飯。")],
        "test": [
            ("我不能喝牛奶。", "給我早餐建議。", "可以選吐司、雞蛋和水，避開牛奶。"),
            ("請叫我阿晴。", "你還記得我希望的稱呼嗎？", "你希望我叫你阿晴。"),
        ],
    }
    for split, examples in dialogs.items():
        for index, (earlier, question, answer) in enumerate(examples):
            family = f"course-dialogue-{split}-{index:02}"
            rows.append(
                {
                    "id": family,
                    "family": family,
                    "split": split,
                    "task": "text_dialogue",
                    "user": question,
                    "answer": answer,
                    "image": None,
                    "history": [
                        {"role": "system", "content": "請用繁體中文簡短回應。"},
                        {"role": "user", "content": earlier},
                        {"role": "assistant", "content": "好，我會記得你的要求。"},
                    ],
                    "references": {"kind": "manual", "text": answer, "constraint": earlier},
                    "source": "course-self-written-chat-MIT",
                }
            )
    return rows


def build(source, assets, manifest_path):
    rows, audio_rows, sources, archives, files = text_rows(), [], [], [], []
    for kind in KINDS:
        directory = source / kind
        document = json.loads((directory / "manifest.json").read_text())
        for field, target in (("rows", rows), ("audio_rows", audio_rows)):
            for original in document.get(field, []):
                row = dict(original)
                row["id"] = f"{kind}:{row['id']}"
                row["family"] = f"{kind}:{row['family']}"
                for key in ("image", "audio"):
                    if row.get(key):
                        relative = Path(row[key])
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError("Dataset contains an unsafe asset path")
                        row[key] = f"{kind}/{relative.as_posix()}"
                if field == "audio_rows":
                    row["history"] = [
                        {"role": "system", "content": "請用繁體中文簡短回應使用者所說的內容，不要只重複原話。"}
                    ]
                target.append(row)
        raw_sources = document.get("sources", [])
        sources.append({"id": kind, "dataset_version": document["dataset_version"], "upstream": raw_sources})
        archive = assets / f"natural-{kind}-v3.tar.gz"
        members = package(directory, archive, kind)
        files.extend(members)
        archives.append(
            {
                "path": f"assets/training/{archive.name}",
                "sha256": sha256(archive),
                "bytes": archive.stat().st_size,
                "files": members,
            }
        )
        print(f"{kind}: {archive.stat().st_size} bytes; {len(members)} files")
    all_rows = rows + audio_rows
    ids = [row["id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated row ID")
    families = {
        name: {row["family"] for row in all_rows if row["split"] == name} for name in ("train", "validation", "test")
    }
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if families[first] & families[second]:
            raise ValueError("A family occurs in multiple splits")
    manifest = {
        "schema_version": 1,
        "dataset_version": VERSION,
        "sources": sources + [{"id": "course-self-written-chat-MIT", "license": "MIT"}],
        "archives": archives,
        "files": files,
        "rows": rows,
        "audio_rows": audio_rows,
        "counts": {
            name: dict(Counter(row["task"] for row in all_rows if row["split"] == name))
            for name in ("train", "validation", "test")
        },
        "scope": "Small published teaching sample, not a comprehensive benchmark. Base pretraining overlap is unknown. Images retain DOCCI CC BY 4.0; FLEURS recordings retain CC BY 4.0; synthetic OCR text is course MIT and source fonts remain OFL. Audio is human read speech, not a dialogue-training corpus. ASR is frozen pretrained; only the visual-language LoRA is trained here.",
        "split_policy": "Keep source image or sentence and all derived questions in one split; assert distinct families; evaluate base and trained variant on identical frozen cases. Exact asset bytes must match files. No claim of unknown-pretraining-free test.",
    }
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/natural")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets/training")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/natural-assistant/manifest.json")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--verify-manifest", type=Path)
    args = parser.parse_args()
    if args.prepare:
        for kind in KINDS:
            subprocess.run(
                [sys.executable, str(ROOT / f"scripts/prepare_natural_{kind}.py"), "--output", str(args.source / kind)],
                check=True,
            )
    actual = build(args.source, args.assets, args.manifest)
    if args.verify_manifest and actual != json.loads(args.verify_manifest.read_text()):
        raise RuntimeError("Reconstructed manifest differs from committed data snapshot")


if __name__ == "__main__":
    main()
