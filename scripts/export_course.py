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


def diagram_html(url, label, diagram_id):
    url, label = html.escape(url, quote=True), html.escape(label, quote=True)
    return (
        f'<figure class="illustration"><div class="diagram-viewport" id="{diagram_id}">'
        f'<a class="diagram-link" href="{url}" target="_blank" rel="noopener">'
        f'<img class="diagram" src="{url}" alt="{label}"></a></div>'
        '<figcaption><span class="diagram-scroll-hint" hidden>可左右滑動看全圖 · </span>'
        f'<button type="button" class="diagram-zoom" aria-expanded="false" '
        f'aria-controls="{diagram_id}" hidden>放大圖解</button> '
        f'<a href="{url}" target="_blank" rel="noopener">開啟原圖 ↗</a></figcaption></figure>'
    )


class ReadingRenderer(mistune.HTMLRenderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.figure_count = 0

    def image(self, text, url, title=None):
        if url.startswith("figures/") and url.endswith(".svg"):
            self.figure_count += 1
            label = html.unescape(re.sub(r"<[^>]+>", "", text))
            return diagram_html(url, label, f"reading-diagram-{self.figure_count}")
        return super().image(text, url, title)

    def paragraph(self, text):
        if text.startswith('<figure class="illustration">') and text.endswith("</figure>"):
            return text + "\n"
        return super().paragraph(text)

    def heading(self, text, level, **attrs):
        plain = html.unescape(re.sub(r"<[^>]+>", "", text))
        lesson = re.match(r"^([A-Z\d]+\.\d+)\s", plain)
        anchor = lesson[1] if lesson else re.sub(r"[^\w\s-]", "", plain.lower()).replace(" ", "-")
        # 指南／名詞頁的編輯追蹤號保留為連結錨點，讀者只需看到題名。
        text = re.sub(r"^[RGT]\.\d+\s+", "", text)
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
        label = (ROOT / items[0]["source"]).read_text(encoding="utf-8").splitlines()[0].removeprefix("# ")
        links = []
        for item in items:
            attribute = ' aria-current="page"' if item["id"] == current else ""
            title = html.escape(item["id"] + " " + item["title"])
            links.append(f'<a href="{item["id"]}.html"{attribute}>{title}</a>')
        opened = " open" if active else ""
        parts.append(f"<details{opened}><summary>{html.escape(label)}</summary>{''.join(links)}</details>")
    return "".join(parts)


def page(title, body, index, current="", revision="main"):
    math_config = json.dumps({"tex": {"inlineMath": [["$", "$"], [r"\(", r"\)"]], "processEscapes": True}})
    math_scripts = (
        f"<script>window.MathJax={math_config};</script>"
        '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>'
        if 'class="math"' in body
        else ""
    )
    asset_revision = html.escape(revision[:12], quote=True)
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="從接字表到小型多模態模型：一次理解一件事，配合圖解與短程式實驗。">
<title>{html.escape(title)} · 小小感知機</title><link rel="stylesheet" href="site.css?v={asset_revision}">
<script defer src="site.js?v={asset_revision}"></script>
{math_scripts}</head><body>
<a class="skip" href="#content">跳到正文</a>
<header><a class="brand" href="index.html">小小感知機</a><span class="badge">教材預覽版</span>
<nav aria-label="主要導覽"><a href="course.html">從這裡開始</a><a href="first-steps.html">基礎暖身與操作</a>
<a href="glossary.html">名詞對照</a><a href="training.html">訓練操作</a></nav></header>
<div class="layout"><aside><details class="toc"><summary>章節目錄</summary>
<nav aria-label="章節與小節"><a href="first-steps.html">基礎暖身：Python、數字表與機率</a>{sidebar(index, current)}</nav></details></aside>
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
    lesson_targets = {(ROOT / item["source"]).resolve(): {} for item in index}
    for item in index:
        lesson_targets[(ROOT / item["source"]).resolve()][item["id"]] = item["id"] + ".html"

    def reading_links(content, source_path):
        def convert(match):
            target = match[2]
            course_url = "https://birdhackor.github.io/tiny-perceptron-vlm/"
            if target.startswith(course_url):
                return "[" + match[1] + "](" + (target[len(course_url) :] or "index.html") + ")"
            if "://" in target:
                return match[0]
            path_text, _, fragment = target.partition("#")
            path = (source_path.parent / path_text).resolve() if path_text else source_path.resolve()
            if fragment in lesson_targets.get(path, {}):
                link = lesson_targets[path][fragment]
                fragment = ""
            elif not path_text:
                return match[0]
            elif path.suffix == ".svg" and path.parent == ROOT / "course/figures":
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
            f'<h1 id="{item["id"]}">{html.escape(title)}</h1>',
        ]
        for number, cell in enumerate(notebook["cells"]):
            source = joined(cell["source"])
            if cell["cell_type"] == "markdown":
                if number == 0:
                    source = re.sub(r"^# [^\n]+\n", "", source, count=1)
                    source = re.sub(r"^# ", "### ", source, flags=re.M)
                content.append(markdown(reading_links(source, ROOT / item["source"])))
            elif "course_figure" in cell.get("metadata", {}):
                visual = cell["metadata"]["course_figure"]
                figure_url = f"figures/{visual['name']}.svg"
                diagram_id = f"diagram-{item['id']}-{number}"
                content.append(diagram_html(figure_url, visual["alt"], diagram_id))
            elif not cell.get("metadata", {}).get("course_setup"):
                content.append('<div class="experiment"><pre><code>' + html.escape(source) + "</code></pre>")
                content.append(output_html(cell.get("outputs", [])))
                content.append("</div>")
        content.append(
            '<div class="actions">'
            f'<a class="button" href="{colab}" target="_blank" rel="noopener">在 Colab 動手做 ↗</a>'
            f'<a class="button secondary" href="{item["notebook"]}" download>下載本節 .ipynb</a></div>'
            '<p class="helper">要試做本節練習，可開啟 Colab；閱讀與練習使用同一份內容。'
            '操作步驟見<a href="first-steps.html#W.1">暖身 W.1</a>，小實驗使用 CPU 即可。</p>'
        )
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
        introduction = re.split(r"^## ", path.read_text(encoding="utf-8"), maxsplit=1, flags=re.M)[0]
        body = markdown(reading_links(introduction, path))
        body += (
            "<h2>本章小節</h2><p>依序閱讀，或點進眼前想弄懂的問題；每節都有具體例子與練習。</p><ul>" + links + "</ul>"
        )
        (destination / target).write_text(page(chapter, body, index, revision=revision), encoding="utf-8")
    body = (
        '<h1>電腦怎麼學會接話？</h1><p class="lead">看到「今天天氣」，你會怎麼接下一個字？'
        "這套教材從這個問題開始，用接字表、具體數字與短程式，逐步解釋語言模型如何學習。"
        "之後再看圖片與聲音怎麼接進同一套系統。</p>"
        '<div class="actions"><a class="button" href="1.1.html">從第一節開始</a>'
        '<a class="button secondary" href="first-steps.html">先看基礎暖身</a></div>'
        "<p>可以先當書讀：程式下方已有實際結果，不必先安裝工具或準備 GPU。"
        "遇到陌生背景，每節都提供對應的前置連結。想先看個性、安全、圖片或量化，"
        '<a href="course.html">閱讀指南</a>會帶你挑需要的幾節。</p>'
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
