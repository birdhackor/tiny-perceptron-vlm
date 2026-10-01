"""由可閱讀的 Markdown 正文產生短 Notebook；--check 檢查是否同步。"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTER_FIGURES = {
    "1": "shift",
    "2": "context",
    "3": "qkv",
    "4": "residual",
    "5": "backward",
    "6": "bpe",
    "7": "masks",
    "8": "style",
    "9": "honesty",
    "10": "patchify",
    "11": "alignment",
    "12": "audio",
    "13": "dpo",
    "14": "rope",
    "15": "moe",
    "16": "cache",
    "17": "quantization",
    "18": "distillation",
    "A": "rag",
    "B": "tools",
    "C": "reasoning",
}
FIGURE_SECTIONS = {
    "lookup": ["1.1", "1.6", "2.1"],
    "split": ["1.2", "5.10", "5.11", "5.12", "7.12"],
    "softmax": ["1.4", "1.5", "1.7", "1.8", "1.15", "3.2", "3.5"],
    "gradient": ["1.9", "1.10", "1.11", "1.12", "5.3", "5.4", "5.5", "5.14"],
    "backward": ["1.13", "5.1", "5.2", "5.17", "16.6"],
    "causal": ["3.6", "3.7", "16.8"],
    "normalization": ["4.3", "14.2", "14.3"],
    "checkpoint": ["5.7"],
    "padding": ["7.6", "7.7", "7.8", "7.10"],
    "lora": ["8.8", "8.9", "8.13"],
    "safety": ["9.1", "9.4", "9.5", "9.6", "9.7", "9.8"],
    "calibration": ["9.9", "9.10"],
    "modal_expand": ["10.6", "10.7", "12.9", "18.13"],
    "contrastive": ["10.9", "10.10", "10.11"],
    "patchify": ["10.1", "10.2", "10.3", "10.4", "11.9", "11.10", "11.12", "11.13"],
    "frames": ["12.1", "12.2", "12.3", "12.4"],
    "joint": ["12.10", "12.11", "12.12"],
    "dispatch": ["15.5", "15.6", "15.7", "15.12"],
    "packing": ["16.5"],
    "online_softmax": ["16.9"],
    "int4": ["17.6", "17.8"],
}
SECTION_FIGURES = {sid: name for name, ids in FIGURE_SECTIONS.items() for sid in ids}
BOOTSTRAP = """from pathlib import Path
import sys

root = Path.cwd().resolve()
while not (root / "pyproject.toml").exists() and root != root.parent:
    root = root.parent
if not (root / "tiny_perceptron").is_dir():
    raise RuntimeError("先依 course/README.md 下載並開啟這個 repo")
sys.path.insert(0, str(root))
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.manual_seed(42)
"""


def cell(kind, source):
    if kind == "code":
        source = source.rstrip("\n")
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook(section_id, title, introduction, body):
    cells = [cell("markdown", f"# {section_id} {title}\n\n{introduction}\n"), cell("code", BOOTSTRAP)]
    figure = SECTION_FIGURES.get(section_id, CHAPTER_FIGURES[section_id.split(".")[0]])
    visual = json.loads((ROOT / "course/figures/index.json").read_text(encoding="utf-8"))[figure]
    hint = (
        "橘色邊框只提示閱讀順序，靜態標註一直保留。系統的減少動態效果設定會停用動畫。"
        if visual["animated"]
        else "這張圖保留靜態標註；先核對箭頭、位置與每個數字的意思。"
    )
    cells += [
        cell(
            "markdown",
            f"**先看圖：{visual['title']}**\n\n{visual['caption']}\n\n{hint}\n",
        ),
        cell(
            "code",
            f'from IPython.display import SVG, display\n\ndisplay(SVG(filename=str(root / "course/figures/{figure}.svg")))\n',
        ),
    ]
    # 正文圖放在 Markdown 閱讀版；Notebook 已在圖解 cell 顯示同一張。
    body = re.sub(r"!\[[^\]]*\]\(\.\./figures/[^)]+\.svg\)\n*", "", body)
    chunks = re.split(r"```python\n(.*?)```", body, flags=re.S)
    for i, text in enumerate(chunks):
        if text.strip():
            cells.append(cell("code" if i % 2 else "markdown", text.strip() + "\n"))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Tiny Perceptron", "language": "python", "name": "tiny-perceptron"},
            "language_info": {"name": "python", "version": "3.13"},
            "lesson_id": section_id,
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def build(check=False):
    index, stale = [], []
    sources = (ROOT / "course/chapters").glob("*.md")
    for source in sorted(sources, key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)):
        text = source.read_text(encoding="utf-8")
        parts = re.split(r"^## ([\dABC]+\.\d+) (.+)$", text, flags=re.M)
        introduction = parts[0]
        for i in range(1, len(parts), 3):
            section_id, title, body = parts[i : i + 3]
            target = ROOT / "notebooks" / source.stem / f"{section_id}.ipynb"
            serialized = (
                json.dumps(notebook(section_id, title, introduction, body), ensure_ascii=False, indent=1) + "\n"
            )
            if check:
                if not target.exists() or target.read_text(encoding="utf-8") != serialized:
                    stale.append(str(target.relative_to(ROOT)))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(serialized, encoding="utf-8")
            index.append(
                {
                    "id": section_id,
                    "title": title,
                    "source": str(source.relative_to(ROOT)),
                    "notebook": str(target.relative_to(ROOT)),
                }
            )
    if not index:
        raise RuntimeError("没有教材來源")
    outline = (ROOT / "docs/curriculum.md").read_text(encoding="utf-8")
    expected = set(re.findall(r"^- ([\dABC]+\.\d+) ", outline, flags=re.M))
    actual = {item["id"] for item in index}
    if expected != actual or len(index) != len(actual):
        raise RuntimeError(f"編號覆蓋不完整：missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    target = ROOT / "course/lesson-index.json"
    serialized = json.dumps(index, ensure_ascii=False, indent=2) + "\n"
    rows = [
        "# 全部小節",
        "",
        "每節可獨立開啟。先讀[操作與數學暖身](first-steps.md)，或從[閱讀路線](README.md)挑一條支線。",
        "",
        "| 小節 | 正文 | Notebook |",
        "| --- | --- | --- |",
    ]
    for item in index:
        rows.append(f"| {item['id']} {item['title']} | [閱讀](../{item['source']}) | [開啟](../{item['notebook']}) |")
    lessons = ROOT / "course/lessons.md"
    lesson_text = "\n".join(rows) + "\n"
    if check:
        if not target.exists() or target.read_text(encoding="utf-8") != serialized:
            stale.append(str(target.relative_to(ROOT)))
        if not lessons.exists() or lessons.read_text(encoding="utf-8") != lesson_text:
            stale.append(str(lessons.relative_to(ROOT)))
        if stale:
            raise RuntimeError(f"正文與 Notebook 不同步：{stale}")
    else:
        target.write_text(serialized, encoding="utf-8")
        lessons.write_text(lesson_text, encoding="utf-8")
    print(f"{len(index)} 個小節：" + ("同步檢查通過" if check else "Notebook 已產生"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    build(p.parse_args().check)
