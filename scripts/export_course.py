"""將教材與核對過的 Notebook 輸出整理成 Markdown，再由 Zensical 建置全站。"""

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "birdhackor/tiny-perceptron-vlm"
COURSE_URL = "https://birdhackor.github.io/tiny-perceptron-vlm/"
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
    "publishing": "docs/publishing.md",
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


def reading_markdown(content, source_path, targets, lesson_targets, revision):
    """只調整發布連結與錨點；正文、程式和圖解仍使用已審閱的來源。"""

    def convert(match):
        target = match[2]
        if target.startswith(COURSE_URL):
            target = target[len(COURSE_URL) :] or "index.html"
            path, separator, fragment = target.partition("#")
            return f"[{match[1]}]({path.removesuffix('.html')}.md{separator}{fragment})"
        if "://" in target or target.startswith("mailto:"):
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
        return f"[{match[1]}]({link}{('#' + fragment) if fragment else ''})"

    figure_count = 0

    def figure(match):
        nonlocal figure_count
        figure_count += 1
        return diagram_html(match[2], match[1], f"reading-diagram-{figure_count}")

    lines = []
    fence = None
    for line in content.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
            lines.append(line)
            continue
        if fence is not None:
            lines.append(line)
            continue
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", convert, line)
        line = re.sub(r"!\[([^\]]*)\]\((figures/[^)]+\.svg)\)", figure, line)
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            title = heading[2]
            # 保留既有 W.2、G.1、1.1 等書籤；指南題名省略編輯追蹤號。
            numbered = re.match(r"^([A-Z\d]+\.\d+)\s", title)
            plain = re.sub(r"[`*_]", "", re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title))
            anchor = numbered[1] if numbered else re.sub(r"[^\w\s-]", "", plain.lower()).replace(" ", "-")
            title = re.sub(r"^[RGT]\.\d+\s+", "", title)
            line = f"{heading[1]} {title} {{#{anchor}}}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def code_markdown(source):
    # 程式字串中即使有 Markdown fence，也不會提早關閉程式區塊。
    fence = "`" * max(3, 1 + max((len(x) for x in re.findall(r"`+", source)), default=0))
    return f"{fence}python\n{source.rstrip()}\n{fence}\n"


def build(destination, executed_root=None, revision="main"):
    configuration = (ROOT / "zensical.toml").read_text(encoding="utf-8")
    settings = tomllib.loads(configuration)["project"]
    docs = ROOT / settings["docs_dir"]
    outputs_root = ROOT / "outputs"
    if docs.resolve() == outputs_root or not docs.resolve().is_relative_to(outputs_root):
        raise ValueError("產生的 Markdown 必須放在 outputs/ 的子目錄")
    if destination == outputs_root or not destination.is_relative_to(outputs_root):
        raise ValueError("Zensical 網站產物必須放在專案 outputs/ 的子目錄")
    if docs.is_relative_to(destination) or destination.is_relative_to(docs):
        raise ValueError("網站目錄與產生的 Markdown 目錄不能重疊")
    index = json.loads((ROOT / "course/lesson-index.json").read_text(encoding="utf-8"))
    targets = {(ROOT / path).resolve(): name + ".md" for name, path in DOCUMENTS.items()}
    targets[(ROOT / "course/lessons.md").resolve()] = "index.md"
    chapters = {item["source"]: "chapter-" + Path(item["source"]).stem + ".md" for item in index}
    for source, target in chapters.items():
        targets[(ROOT / source).resolve()] = target
    lesson_targets = {(ROOT / item["source"]).resolve(): {} for item in index}
    for item in index:
        targets[(ROOT / item["notebook"]).resolve()] = item["id"] + ".md"
        lesson_targets[(ROOT / item["source"]).resolve()][item["id"]] = item["id"] + ".md"

    def reading(content, source):
        return reading_markdown(content, source, targets, lesson_targets, revision)

    # 先核對全部執行副本，再變更產物；舊輸出不能混入新網站。
    notebooks = []
    for item in index:
        notebook = json.loads((ROOT / item["notebook"]).read_text(encoding="utf-8"))
        if executed_root:
            notebook = checked_notebook(notebook, executed_root / Path(item["notebook"]).relative_to("notebooks"))
        notebooks.append(notebook)
    if docs.exists():
        shutil.rmtree(docs)
    docs.mkdir(parents=True)
    shutil.copytree(ROOT / "course/figures", docs / "figures")
    shutil.copytree(ROOT / "notebooks", docs / "notebooks")
    for folder, names in {"stylesheets": ["course.css"], "javascripts": ["course.js", "mathjax.js"]}.items():
        (docs / folder).mkdir()
        for name in names:
            shutil.copyfile(ROOT / "course/web" / name, docs / folder / name)

    # 發布引用依內容命名的樣式，讓瀏覽器載入修正版；原檔留給 zensical serve。
    stylesheet = docs / "stylesheets/course.css"
    style_hash = hashlib.sha256(stylesheet.read_bytes()).hexdigest()[:12]
    style_url = f"stylesheets/course.{style_hash}.css"
    shutil.copyfile(stylesheet, docs / style_url)
    configuration = configuration.replace('"stylesheets/course.css"', json.dumps(style_url))

    for item, notebook in zip(index, notebooks, strict=True):
        content = [f"# {item['id']} {item['title']} {{#{item['id']}}}"]
        for number, cell in enumerate(notebook["cells"]):
            source = joined(cell["source"])
            if cell["cell_type"] == "markdown":
                if number == 0:
                    source = re.sub(r"^# [^\n]+\n?", "", source, count=1)
                content.append(reading(source, ROOT / item["source"]))
            elif "course_figure" in cell.get("metadata", {}):
                visual = cell["metadata"]["course_figure"]
                content.append(
                    diagram_html(f"figures/{visual['name']}.svg", visual["alt"], f"diagram-{item['id']}-{number}")
                )
            elif not cell.get("metadata", {}).get("course_setup"):
                content.append(code_markdown(source))
                content.append(output_html(cell.get("outputs", [])))
        colab = f"https://colab.research.google.com/github/{REPOSITORY}/blob/{revision}/{item['notebook']}"
        content.append(
            '<div class="actions">'
            f'<a class="md-button md-button--primary" href="{html.escape(colab, quote=True)}" '
            'target="_blank" rel="noopener">在 Colab 動手做 ↗</a>'
            f'<a class="md-button" href="{item["notebook"]}" download>下載本節 .ipynb</a></div>\n\n'
            "要試做本節練習，可開啟 Colab；閱讀與練習使用同一份內容。"
            "操作步驟見[暖身 W.1](first-steps.md#W.1)，小實驗使用 CPU 即可。"
        )
        (docs / f"{item['id']}.md").write_text("\n\n".join(content) + "\n", encoding="utf-8")
    for name, relative in DOCUMENTS.items():
        path = ROOT / relative
        (docs / f"{name}.md").write_text(reading(path.read_text(encoding="utf-8"), path), encoding="utf-8")
    for source, target in chapters.items():
        path = ROOT / source
        items = [item for item in index if item["source"] == source]
        introduction = re.split(r"^## ", path.read_text(encoding="utf-8"), maxsplit=1, flags=re.M)[0]
        content = reading(introduction, path) + "\n## 本章小節\n\n依序閱讀，或點進眼前想弄懂的問題。\n\n"
        content += "\n".join(f"- [{item['id']} {item['title']}]({item['id']}.md)" for item in items) + "\n"
        (docs / target).write_text(content, encoding="utf-8")

    home = (
        "# 電腦怎麼學會接話？\n\n"
        "看到「今天天氣」，你會怎麼接下一個字？這套教材從這個問題開始，"
        "用接字表、具體數字與短程式，逐步解釋語言模型如何學習。"
        "之後再看圖片與聲音怎麼接進同一套系統。\n\n"
        "[從第一節開始](1.1.md){ .md-button .md-button--primary }\n"
        "[先看基礎暖身](first-steps.md){ .md-button }\n\n"
        + (
            "可以先當書讀：程式下方已有實際結果，不必先安裝工具或準備 GPU。"
            if executed_root
            else "可以先當書讀：這份建置包含正文與程式，尚未附上執行結果。"
        )
        + "遇到陌生背景，每節都提供對應的前置連結。想先看個性、安全、圖片或量化，"
        "[閱讀指南](course.md)會帶你挑需要的幾節。\n\n"
        "## 找到想讀的內容\n\n"
        f"上方搜尋可以找小節標題和正文中的概念。章節目錄收錄全部 {len(index)} 個小節；"
        "手機上可點左上角的選單開啟目錄。每節下方都有 Colab 入口與 Notebook 下載，"
        "想動手改程式時再開啟即可。\n\n"
        "## 章節\n\n"
    )
    for source, target in chapters.items():
        title = (ROOT / source).read_text(encoding="utf-8").splitlines()[0].removeprefix("# ")
        home += f"- [{title}]({target})\n"
    (docs / "index.md").write_text(home, encoding="utf-8")

    def nav_paths(items):
        for entry in items:
            if isinstance(entry, str):
                yield entry
            else:
                for value in entry.values():
                    yield from nav_paths(value) if isinstance(value, list) else [value]

    expected = {path.name for path in docs.glob("*.md")}
    actual = list(nav_paths(settings["nav"]))
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("zensical.toml 導覽與教材頁面不一致，請依 lesson-index.json 更新導覽")
    # Zensical 以設定檔所在目錄作為專案根目錄；網站不能寫到根目錄外。
    config_path = ROOT / ".zensical-build.toml"
    configuration = re.sub(
        r"^docs_dir = .*$", "docs_dir = " + json.dumps(docs.relative_to(ROOT).as_posix()), configuration, flags=re.M
    )
    configuration = re.sub(
        r"^site_dir = .*$",
        "site_dir = " + json.dumps(destination.relative_to(ROOT).as_posix()),
        configuration,
        flags=re.M,
    )
    config_path.write_text(configuration, encoding="utf-8")
    try:
        subprocess.run(
            [sys.executable, "-m", "zensical", "build", "--clean", "--strict", "--config-file", str(config_path)],
            cwd=ROOT,
            check=True,
        )
    finally:
        config_path.unlink()
    (destination / ".nojekyll").write_text("", encoding="utf-8")
    (destination / "build-info.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "lessons": len(index),
                "executed_cpu_outputs": bool(executed_root),
                "builder": "zensical",
                "builder_version": version("zensical"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Zensical 建置 {len(index)} 節至 {destination}；CPU 輸出：{'已核對' if executed_root else '未提供'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/site")
    p.add_argument("--executed", type=Path, help="獨立 kernel 執行副本；提供時嚴格核對每份教材與執行結果")
    p.add_argument("--revision", default="main", help="網站原始碼與 Colab 連結的 Git ref")
    args = p.parse_args()
    build(args.output.resolve(), args.executed, args.revision)
