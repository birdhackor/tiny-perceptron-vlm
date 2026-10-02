"""離線檢查發布網站的內部連結、圖檔、Colab 入口與每節 Notebook 下載。"""

import argparse
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
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
    pages = {}
    for path in site.glob("*.html"):
        parser = Links()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = parser
    failures = []
    local_links = 0
    for path, parsed in pages.items():
        for link in parsed.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            local_links += 1
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.is_relative_to(site.resolve()) or not target.is_file():
                failures.append(f"{path.name}: 缺少 {link}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                failures.append(f"{path.name}: 缺少段落 {link}")
    for item in index:
        path = (site / (item["id"] + ".html")).resolve()
        if path not in pages:
            failures.append(f"缺少小節 {item['id']}")
            continue
        notebook_path = site / item["notebook"]
        if item["notebook"] not in pages[path].links or not notebook_path.is_file():
            failures.append(f"缺少下載 {item['id']}")
        else:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            if notebook.get("metadata", {}).get("lesson_id") != item["id"]:
                failures.append(f"下載了不同小節的 Notebook：{item['id']}")
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
    print(f"{len(pages)} 頁、{len(index)} 份下載與 Colab 入口、{local_links} 個內部連結通過")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", type=Path, default=ROOT / "outputs/site")
    check(p.parse_args().site.resolve())
