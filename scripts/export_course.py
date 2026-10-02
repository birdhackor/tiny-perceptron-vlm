"""匯出教材、已驗證的 CPU 輸出與 Notebook；網站產物只寫入 ignored outputs/site。"""

import argparse
import html
import json
import re
import shutil
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "birdhackor/tiny-perceptron-vlm"
DOCUMENTS = {
    "course": "course/README.md",
    "first-steps": "course/first-steps.md",
    "training": "course/training.md",
    "glossary": "course/glossary.md",
    "readme": "README.md",
    "environment": "docs/environment.md",
    "asset-storage": "docs/asset-storage.md",
    "curriculum": "docs/curriculum.md",
    "validation": "docs/validation.md",
    "training-assets": "assets/training/README.md",
}


class ReadingRenderer(mistune.HTMLRenderer):
    def heading(self, text, level, **attrs):
        plain = html.unescape(re.sub(r"<[^>]+>", "", text)).lower()
        anchor = re.sub(r"[^\w\s-]", "", plain).replace(" ", "-")
        return f'<h{level} id="{html.escape(anchor, quote=True)}">{text}</h{level}>\n'


def joined(value):
    return "".join(value) if isinstance(value, list) else value


def output_html(outputs):
    parts = []
    for output in outputs:
        kind = output["output_type"]
        if kind == "error":
            raise ValueError("有錯誤的 Notebook 不能當作成功範例發布")
        if kind == "stream":
            parts.append("<pre>" + html.escape(joined(output["text"])) + "</pre>")
        elif kind in ("display_data", "execute_result"):
            data = output["data"]
            if "image/png" in data:
                encoded = html.escape(joined(data["image/png"]), quote=True)
                parts.append(f'<img src="data:image/png;base64,{encoded}" alt="本段程式的實際輸出圖">')
            elif "text/plain" in data:
                parts.append("<pre>" + html.escape(joined(data["text/plain"])) + "</pre>")
    if not parts:
        return ""
    return '<div class="output"><p class="output-label">實際執行結果 · CPU</p>' + "".join(parts) + "</div>"


def checked_notebook(original, executed_path):
    executed = json.loads(executed_path.read_text(encoding="utf-8"))
    if len(original["cells"]) != len(executed["cells"]):
        raise ValueError(f"執行副本的 cell 數量不同：{executed_path}")
    for source, result in zip(original["cells"], executed["cells"], strict=True):
        if source["cell_type"] != result["cell_type"] or joined(source["source"]) != joined(result["source"]):
            raise ValueError(f"執行結果不是目前的教材：{executed_path}")
        if result["cell_type"] == "code":
            if result.get("execution_count") is None:
                raise ValueError(f"程式尚未執行：{executed_path}")
            if any(output["output_type"] == "error" for output in result.get("outputs", [])):
                raise ValueError(f"程式執行失敗：{executed_path}")
    return executed


def sidebar(index, current):
    groups = {}
    for item in index:
        groups.setdefault(item["id"].split(".")[0], []).append(item)
    parts = []
    for chapter, items in groups.items():
        active = any(item["id"] == current for item in items)
        label = f"第 {chapter} 章" if chapter.isdigit() else f"支線 {chapter}"
        links = []
        for item in items:
            attribute = ' aria-current="page"' if item["id"] == current else ""
            title = html.escape(item["id"] + " " + item["title"])
            links.append(f'<a href="{item["id"]}.html"{attribute}>{title}</a>')
        opened = " open" if active else ""
        parts.append(f"<details{opened}><summary>{label}</summary>{''.join(links)}</details>")
    return "".join(parts)


def page(title, body, index, current="", revision="main"):
    math_config = json.dumps({"tex": {"inlineMath": [["$", "$"], [r"\(", r"\)"]], "processEscapes": True}})
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="從接字表到小型多模態模型：一次理解一件事，配合圖解與短程式實驗。">
<title>{html.escape(title)} · 小小感知機</title><link rel="stylesheet" href="site.css">
<script>window.MathJax={math_config};</script>
<script defer src="site.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script></head><body>
<a class="skip" href="#content">跳到正文</a>
<header><a class="brand" href="index.html">小小感知機</a><span class="badge">教材預覽版</span>
<nav aria-label="主要導覽"><a href="course.html">閱讀路線</a><a href="first-steps.html">開始練習</a>
<a href="glossary.html">名詞對照</a><a href="training.html">訓練操作</a></nav></header>
<div class="layout"><aside><details class="toc"><summary>章節目錄</summary>
<nav aria-label="章節與小節">{sidebar(index, current)}</nav></details></aside>
<main id="content">{body}</main></div>
<footer>小型數值實驗已在 CPU 驗證；正式 GPU 訓練與模型能力尚待實測。
<a href="validation.html">驗證範圍</a> · <a href="https://github.com/{REPOSITORY}">GitHub</a>
<span class="revision">教材版本 {html.escape(revision[:12])}</span></footer></body></html>"""


def build(destination, executed_root=None, revision="main"):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "course/figures", destination / "figures", dirs_exist_ok=True)
    for name in ("site.css", "site.js"):
        shutil.copyfile(ROOT / "course" / name, destination / name)
    shutil.copytree(ROOT / "notebooks", destination / "notebooks", dirs_exist_ok=True)
    index = json.loads((ROOT / "course/lesson-index.json").read_text(encoding="utf-8"))
    markdown = mistune.create_markdown(renderer=ReadingRenderer(escape=True), plugins=["table", "math"])
    targets = {(ROOT / path).resolve(): name + ".html" for name, path in DOCUMENTS.items()}
    targets[(ROOT / "course/lessons.md").resolve()] = "index.html"
    chapters = {item["source"]: "chapter-" + Path(item["source"]).stem + ".html" for item in index}
    for source, target in chapters.items():
        targets[(ROOT / source).resolve()] = target
    for item in index:
        targets[(ROOT / item["notebook"]).resolve()] = item["id"] + ".html"

    def reading_links(content, source_path):
        def convert(match):
            target = match[2]
            if "://" in target or target.startswith("#"):
                return match[0]
            path_text, _, fragment = target.partition("#")
            path = (source_path.parent / path_text).resolve()
            if path.suffix == ".svg" and path.parent == ROOT / "course/figures":
                link = "figures/" + path.name
            elif path in targets:
                link = targets[path]
            else:
                try:
                    relative = path.relative_to(ROOT)
                except ValueError:
                    return match[0]
                link = f"https://github.com/{REPOSITORY}/blob/{revision}/" + relative.as_posix()
            return "[" + match[1] + "](" + link + (("#" + fragment) if fragment else "") + ")"

        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", convert, content)

    cards = []
    for position, item in enumerate(index):
        original = json.loads((ROOT / item["notebook"]).read_text(encoding="utf-8"))
        notebook = original
        if executed_root:
            notebook = checked_notebook(original, executed_root / Path(item["notebook"]).relative_to("notebooks"))
        colab = f"https://colab.research.google.com/github/{REPOSITORY}/blob/{revision}/{item['notebook']}"
        title = item["id"] + " " + item["title"]
        content = [
            f"<h1>{html.escape(title)}</h1>",
            '<div class="actions">'
            f'<a class="button" href="{colab}" target="_blank" rel="noopener">在 Colab 動手做 ↗</a>'
            f'<a class="button secondary" href="{item["notebook"]}" download>下載本節 .ipynb</a></div>',
            '<p class="helper">Colab 會開啟同一課的說明與程式；登入 Google 後，選擇「執行階段 → 全部執行」。'
            "第一個程式格會下載專案並安裝套件。小實驗使用 CPU 即可。"
            '下載後在本機執行，請先看<a href="first-steps.html">開始練習</a>。</p>',
        ]
        for number, cell in enumerate(notebook["cells"]):
            source = joined(cell["source"])
            if cell["cell_type"] == "markdown":
                if number == 0:
                    source = re.sub(r"^# [^\n]+\n", "", source, count=1)
                    source = re.sub(r"^# ", "### ", source, flags=re.M)
                content.append(markdown(reading_links(source, ROOT / item["source"])))
            elif "IPython.display import SVG" in source:
                figure = source.split("course/figures/")[1].split(".svg")[0]
                content.append(f'<img class="diagram" src="figures/{figure}.svg" alt="本節圖解">')
            elif "sys.path.insert" not in source:
                content.append('<div class="experiment"><pre><code>' + html.escape(source) + "</code></pre>")
                content.append(output_html(cell.get("outputs", [])))
                content.append("</div>")
        neighbors = []
        for offset, label in ((-1, "上一節"), (1, "下一節")):
            if 0 <= position + offset < len(index):
                neighbor = index[position + offset]
                text = html.escape(neighbor["id"] + " " + neighbor["title"])
                neighbors.append(f'<a href="{neighbor["id"]}.html">{label}：{text}</a>')
        content.append('<nav class="neighbors" aria-label="相鄰小節">' + "".join(neighbors) + "</nav>")
        (destination / f"{item['id']}.html").write_text(
            page(title, "".join(content), index, item["id"], revision), encoding="utf-8"
        )
        cards.append(f'<li class="lesson"><a href="{item["id"]}.html">{html.escape(title)}</a></li>')
    for name, relative in DOCUMENTS.items():
        path = ROOT / relative
        content = reading_links(path.read_text(encoding="utf-8"), path)
        (destination / f"{name}.html").write_text(
            page(name, markdown(content), index, revision=revision), encoding="utf-8"
        )
    for source, target in chapters.items():
        path = ROOT / source
        chapter = Path(source).stem
        chapter_items = [item for item in index if Path(item["source"]).stem == chapter]
        links = "".join(
            f'<li><a href="{item["id"]}.html">{html.escape(item["id"] + " " + item["title"])}</a></li>'
            for item in chapter_items
        )
        body = "<p>選擇小節，可看實際輸出並開啟對應練習。</p><ul>" + links + "</ul>"
        body += markdown(reading_links(path.read_text(encoding="utf-8"), path))
        (destination / target).write_text(page(chapter, body, index, revision=revision), encoding="utf-8")
    body = (
        '<h1>一次看懂一件事</h1><p class="lead">從一張接字表開始，慢慢看懂能接收文字、圖片與聲音的小模型。'
        "每次只學一小段：先看圖與譬喻，再讀短程式與實際結果。</p>"
        '<div class="actions"><a class="button" href="1.1.html">從第一節開始</a>'
        '<a class="button secondary" href="course.html">挑選閱讀路線</a></div>'
        "<p>想試著改數字？每節都有「在 Colab 動手做」按鈕，會開啟同一課的完整內容。"
        "先用 CPU 做小實驗；不必先準備 GPU。</p>"
        '<label for="search">搜尋全部 222 個小節</label>'
        '<input id="search" type="search" placeholder="例如：圖片、3.6、量化" autocomplete="off">'
        '<p id="search-count" aria-live="polite">222 個小節</p><ul class="lessons">' + "".join(cards) + "</ul>"
    )
    (destination / "index.html").write_text(page("一次看懂一件事", body, index, revision=revision), encoding="utf-8")
    (destination / ".nojekyll").write_text("", encoding="utf-8")
    (destination / "build-info.json").write_text(
        json.dumps({"revision": revision, "lessons": len(index), "executed_cpu_outputs": bool(executed_root)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"匯出 {len(index)} 節至 {destination}；CPU 輸出：{'已核對' if executed_root else '未提供'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/site")
    p.add_argument("--executed", type=Path, help="獨立 kernel 執行副本；提供時嚴格核對每份教材與執行結果")
    p.add_argument("--revision", default="main", help="網站原始碼與 Colab 連結的 Git ref")
    args = p.parse_args()
    build(args.output.resolve(), args.executed, args.revision)
