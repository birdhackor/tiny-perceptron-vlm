"""把短 Notebook 匯出成可離線閱讀的網頁；只寫入 ignored outputs/site。"""

import argparse
import html
import json
import re
import shutil
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[1]


def page(title, body):
    return f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{max-width:900px;margin:2rem auto;padding:0 1rem;color:#172b4d;font:18px/1.8 system-ui,sans-serif}}
a{{color:#145cab}}pre{{overflow:auto;padding:1rem;background:#f0f4f8;font-size:15px;line-height:1.6}}
img{{max-width:100%;height:auto}}input{{font:inherit;padding:.6rem;width:90%}}.lesson{{margin:.4rem 0}}</style>
<nav><a href="index.html">所有小節</a> · <a href="first-steps.html">操作與數學暖身</a> · <a href="training.html">訓練操作</a></nav>
{body}</html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/site")
    args = p.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "course/figures", destination / "figures", dirs_exist_ok=True)
    index = json.loads((ROOT / "course/lesson-index.json").read_text(encoding="utf-8"))
    markdown = mistune.create_markdown(escape=True)
    documents = {
        "course": "course/README.md",
        "first-steps": "course/first-steps.md",
        "training": "course/training.md",
        "glossary": "course/glossary.md",
        "readme": "README.md",
        "environment": "docs/environment.md",
        "asset-storage": "docs/asset-storage.md",
        "curriculum": "docs/curriculum.md",
    }
    document_targets = {(ROOT / path).resolve(): name + ".html" for name, path in documents.items()}
    document_targets[(ROOT / "course/lessons.md").resolve()] = "index.html"
    for item in index:
        document_targets.setdefault((ROOT / item["source"]).resolve(), item["id"] + ".html")
        document_targets[(ROOT / item["notebook"]).resolve()] = item["id"] + ".html"

    def reading_links(content, source_path):
        def convert(match):
            target = match[2]
            if "://" in target or target.startswith("#"):
                return match[0]
            path_text, _, fragment = target.partition("#")
            path = (source_path.parent / path_text).resolve()
            if path.suffix == ".svg" and path.parent == ROOT / "course/figures":
                destination = "figures/" + path.name
            elif path in document_targets:
                destination = document_targets[path]
            else:
                try:
                    relative = path.relative_to(ROOT)
                except ValueError:
                    return match[0]
                destination = "https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/" + relative.as_posix()
            return "[" + match[1] + "](" + destination + (("#" + fragment) if fragment else "") + ")"

        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", convert, content)

    links = []
    for item in index:
        notebook = json.loads((ROOT / item["notebook"]).read_text(encoding="utf-8"))
        content = []
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                content.append(markdown(reading_links(source, ROOT / item["source"])))
            elif "IPython.display import SVG" in source:
                figure = source.split("course/figures/")[1].split(".svg")[0]
                content.append(f'<img src="figures/{figure}.svg" alt="本節圖解">')
            elif "sys.path.insert" not in source:
                content.append("<pre><code>" + html.escape(source) + "</code></pre>")
        title = item["id"] + " " + item["title"]
        (destination / f"{item['id']}.html").write_text(page(title, "".join(content)), encoding="utf-8")
        links.append(f'<li class="lesson"><a href="{item["id"]}.html">{html.escape(title)}</a></li>')
    for name, relative in documents.items():
        path = ROOT / relative
        content = reading_links(path.read_text(encoding="utf-8"), path)
        (destination / f"{name}.html").write_text(page(name, markdown(content)), encoding="utf-8")
    body = (
        """<h1>小小感知機：一次看懂一件事</h1><p>網頁用來讀圖與程式；要改數字、執行與看結果，開啟對應 Notebook。</p>
<label for="search">搜尋小節編號或標題</label><input id="search" type="search" placeholder="例如：圖片、3.6、量化">
<ul>"""
        + "".join(links)
        + """</ul><script>
document.getElementById('search').addEventListener('input',function(){
const query=this.value.trim().toLowerCase();document.querySelectorAll('.lesson').forEach(function(item){
item.hidden=!item.textContent.toLowerCase().includes(query);});});</script>"""
    )
    (destination / "index.html").write_text(page("小小感知機", body), encoding="utf-8")
    print(f"匯出 {len(index)} 節至 {destination}；打開 index.html 即可離線閱讀")


if __name__ == "__main__":
    main()
