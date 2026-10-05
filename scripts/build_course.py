"""由可閱讀的 Markdown 正文產生短 Notebook；--check 檢查是否同步。"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COURSE_URL = "https://birdhackor.github.io/tiny-perceptron-vlm/"
READING_PAGES = {
    "course/README.md": "course.html",
    "course/first-steps.md": "first-steps.html",
    "course/training.md": "training.html",
    "course/glossary.md": "glossary.html",
    "course/lessons.md": "index.html",
    "README.md": "readme.html",
    "docs/environment.md": "environment.html",
    "docs/asset-storage.md": "asset-storage.html",
    "docs/curriculum.md": "curriculum.html",
    "docs/validation.md": "validation.html",
    "assets/training/README.md": "training-assets.html",
    "docs/natural-assistant/v4/STUDENT.md": "natural-v4-student.html",
    "docs/natural-assistant/v4/DATA.md": "natural-v4-data.html",
    "docs/natural-assistant/v4/TRAINING.md": "natural-v4-training.html",
}
BOOTSTRAP = """# @title 準備本節的工具（首次執行）
from pathlib import Path
import os
import subprocess
import sys

# Colab 會先下載專案；本機 Jupyter 沿用已開啟的 repo。
try:
    import google.colab
except ImportError:
    in_colab = False
else:
    in_colab = True

if in_colab:
    root = Path("/content/tiny-perceptron-vlm")
    if not root.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/birdhackor/tiny-perceptron-vlm.git", str(root)],
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
            check=True,
        )
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-e", str(root)], check=True)
    os.chdir(root)

root = Path.cwd().resolve()
while not (root / "pyproject.toml").exists() and root != root.parent:
    root = root.parent
if not (root / "tiny_perceptron").is_dir():
    raise RuntimeError("本機請先依 course/first-steps.md 開啟 repo；免安裝練習可使用本節的 Colab 入口")
sys.path.insert(0, str(root))
import torch

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


def lesson_anchor_aliases(source):
    """Map a full numbered-heading slug to its preserved lesson ID."""
    if not source.is_file():
        return {}
    aliases = {}
    for lesson, title in re.findall(r"^## ([\dABC]+\.\d+) (.+)$", source.read_text(encoding="utf-8"), re.M):
        heading = f"{lesson} {title}"
        plain = re.sub(r"[`*_]", "", re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading))
        slug = re.sub(r"[^\w\s-]", "", plain.lower()).replace(" ", "-")
        aliases[slug] = lesson
    return aliases


def notebook_reading_links(content, source):
    # Notebook 位於 notebooks/，原稿位於 course/；前置統一接到可閱讀的網站頁面。
    def convert(match):
        target = match[2]
        if "://" in target:
            return match[0]
        path_text, _, fragment = target.partition("#")
        path = (source.parent / path_text).resolve() if path_text else source.resolve()
        if path.parent == ROOT / "course/chapters":
            if fragment and not re.fullmatch(r"[\dABC]+\.\d+", fragment):
                fragment = lesson_anchor_aliases(path).get(fragment, fragment)
        if not path_text and not re.fullmatch(r"[\dABC]+\.\d+", fragment):
            return match[0]
        if path.parent == ROOT / "course/chapters":
            page = fragment + ".html" if re.fullmatch(r"[\dABC]+\.\d+", fragment) else f"chapter-{path.stem}.html"
            if re.fullmatch(r"[\dABC]+\.\d+", fragment):
                fragment = ""
        else:
            try:
                relative = path.relative_to(ROOT).as_posix()
            except ValueError:
                return match[0]
            if relative not in READING_PAGES:
                if path_text and path.is_file():
                    suffix = ("#" + fragment) if fragment else ""
                    return (
                        f"[{match[1]}](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/{relative}{suffix})"
                    )
                return match[0]
            page = READING_PAGES[relative]
        url = COURSE_URL + page + (("#" + fragment) if fragment else "")
        return f"[{match[1]}]({url})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", convert, content)


def notebook(section_id, title, body, source):
    # 只呈現作者寫好的正文。圖解保持原來位置，不自動插入整章摘要或操作提示。
    cells = [cell("markdown", f"# {section_id} {title}\n")]
    blocks = re.compile(
        r"```python\n(?P<python>.*?)```|!\[(?P<alt>[^\]]*)\]\(\.\./figures/(?P<figure>[^/)]+)\.svg\)", re.S
    )
    cursor = 0
    for match in blocks.finditer(body):
        prose = body[cursor : match.start()].strip()
        if prose:
            cells.append(cell("markdown", notebook_reading_links(prose, source) + "\n"))
        if not any(c.get("metadata", {}).get("course_setup") for c in cells):
            setup = cell("code", BOOTSTRAP)
            setup["metadata"] = {
                "course_setup": True,
                "tags": ["hide-input"],
                "jupyter": {"source_hidden": True},
                "cellView": "form",
            }
            cells.append(setup)
        if match["python"] is not None:
            cells.append(cell("code", match["python"].strip() + "\n"))
        else:
            figure = match["figure"]
            if not (ROOT / "course/figures" / f"{figure}.svg").is_file():
                raise ValueError(f"{section_id} 的圖不存在：{figure}")
            diagram = cell(
                "code",
                f'# @title 本節圖解\nfrom IPython.display import SVG, display\n\ndisplay(SVG(filename=str(root / "course/figures/{figure}.svg")))\n',
            )
            diagram["metadata"] = {
                "course_figure": {"name": figure, "alt": match["alt"]},
                "tags": ["hide-input"],
                "jupyter": {"source_hidden": True},
                "cellView": "form",
            }
            cells.append(diagram)
        cursor = match.end()
    prose = body[cursor:].strip()
    if prose:
        cells.append(cell("markdown", notebook_reading_links(prose, source) + "\n"))
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
        for i in range(1, len(parts), 3):
            section_id, title, body = parts[i : i + 3]
            target = ROOT / "notebooks" / source.stem / f"{section_id}.ipynb"
            serialized = json.dumps(notebook(section_id, title, body, source), ensure_ascii=False, indent=1) + "\n"
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
        raise RuntimeError("沒有教材來源")
    contract = json.loads((ROOT / "course/lesson-contract.json").read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or not isinstance(contract.get("lesson_ids"), list):
        raise RuntimeError("小節編號清單格式不正確")
    expected = set(contract["lesson_ids"])
    if len(expected) != len(contract["lesson_ids"]):
        raise RuntimeError("小節編號清單有重複")
    actual = {item["id"] for item in index}
    if expected != actual or len(index) != len(actual):
        raise RuntimeError(f"編號覆蓋不完整：missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    target = ROOT / "course/lesson-index.json"
    serialized = json.dumps(index, ensure_ascii=False, indent=2) + "\n"
    rows = [
        "# 全部小節",
        "",
        "每節聚焦一個問題，網站與Notebook使用同一份正文。需要時查[基礎暖身](first-steps.md)，也可從[閱讀指南](README.md)挑一條路線。",
        "",
        "| 小節 | 正文 | Notebook |",
        "| --- | --- | --- |",
    ]
    for item in index:
        rows.append(
            f"| {item['id']} {item['title']} | [閱讀](../{item['source']}#{item['id']}) | [開啟](../{item['notebook']}) |"
        )
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
