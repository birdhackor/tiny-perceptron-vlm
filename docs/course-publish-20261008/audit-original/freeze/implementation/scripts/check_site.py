"""離線檢查發布網站的內部連結、圖檔、Colab 入口與每節 Notebook 下載。"""

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.generator = ""

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "meta" and attrs.get("name") == "generator":
            self.generator = attrs.get("content", "")
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag in ("a", "link") and "href" in attrs:
            self.links.append(attrs["href"])
        if tag in ("img", "script") and "src" in attrs:
            self.links.append(attrs["src"])


def check(site):
    info = json.loads((site / "build-info.json").read_text(encoding="utf-8"))
    index = json.loads((ROOT / "course/lesson-index.json").read_text(encoding="utf-8"))
    if info["lessons"] != len(index):
        raise ValueError("網站節數與教材不一致")
    if info.get("builder") != "zensical":
        raise ValueError("網站必須使用 Zensical 建置")
    search = json.loads((site / "search.json").read_text(encoding="utf-8"))
    search_pages = {item["location"].split("#")[0] for item in search["items"]}
    pages = {}
    for path in site.glob("*.html"):
        parser = Links()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = parser
    failures = []
    local_links = 0
    notebook_links = 0
    site_prefix = "/tiny-perceptron-vlm/"
    for path, parsed in pages.items():
        if parsed.generator != "zensical-" + info["builder_version"]:
            failures.append(f"{path.name}: 不是目前的 Zensical 建置頁面")
        for link in parsed.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            local_links += 1
            link_path = unquote(url.path)
            if link_path.startswith(site_prefix):
                target = (site / link_path.removeprefix(site_prefix)).resolve()
            else:
                target = (path.parent / link_path).resolve() if link_path else path
            if target.is_dir():
                target = target / "index.html"
            if not target.is_relative_to(site.resolve()) or not target.is_file():
                failures.append(f"{path.name}: 缺少 {link}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                failures.append(f"{path.name}: 缺少段落 {link}")
    for item in index:
        path = (site / (item["id"] + ".html")).resolve()
        if path not in pages:
            failures.append(f"缺少小節 {item['id']}")
            continue
        if item["id"] not in pages[path].ids:
            failures.append(f"小節書籤遺失：{item['id']}")
        if item["id"] + ".html" not in search_pages:
            failures.append(f"全文搜尋未收錄：{item['id']}")
        notebook_path = site / item["notebook"]
        if item["notebook"] not in pages[path].links or not notebook_path.is_file():
            failures.append(f"缺少下載 {item['id']}")
        else:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            original = json.loads((ROOT / item["notebook"]).read_text(encoding="utf-8"))
            if notebook != original:
                failures.append(f"下載與目前教材不同：{item['id']}")
            visible_code = sum(
                cell["cell_type"] == "code"
                and not cell.get("metadata", {}).get("course_setup")
                and "course_figure" not in cell.get("metadata", {})
                for cell in original["cells"]
            )
            page_text = path.read_text(encoding="utf-8")
            if page_text.count('class="language-python highlight"') < visible_code:
                failures.append(f"程式區塊遺失：{item['id']}")
            if notebook.get("metadata", {}).get("lesson_id") != item["id"]:
                failures.append(f"下載了不同小節的 Notebook：{item['id']}")
            for cell in notebook["cells"]:
                if cell["cell_type"] != "markdown":
                    continue
                for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", "".join(cell["source"])):
                    prefix = "https://birdhackor.github.io/tiny-perceptron-vlm/"
                    if link.startswith(prefix):
                        notebook_links += 1
                        url = urlsplit(link[len(prefix) :])
                        target = (site / unquote(url.path)).resolve()
                        if target not in pages or (url.fragment and unquote(url.fragment) not in pages[target].ids):
                            failures.append(f"{item['id']}: Notebook 前置缺頁面或段落 {link}")
                    elif not urlsplit(link).scheme and ".md" in link:
                        failures.append(f"{item['id']}: Notebook 殘留原稿路徑 {link}")
        colab = (
            "https://colab.research.google.com/github/birdhackor/tiny-perceptron-vlm/blob/"
            + info["revision"]
            + "/"
            + item["notebook"]
        )
        if colab not in pages[path].links:
            failures.append(f"Colab 入口版本不符：{item['id']}")
    if failures:
        raise ValueError("\n".join(failures))
    print(
        f"{len(pages)} 頁、{len(index)} 份下載與 Colab 入口、{local_links} 個內部連結、"
        f"{notebook_links} 個 Notebook 閱讀連結通過"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", type=Path, default=ROOT / "outputs/site")
    check(p.parse_args().site.resolve())
