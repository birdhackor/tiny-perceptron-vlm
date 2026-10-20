"""唯讀核對首頁介紹、教材清單及公開網站，僅將摘要輸出到標準輸出。"""

import concurrent.futures
import hashlib
import html
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts import export_course

BASE = "https://birdhackor.github.io/tiny-perceptron-vlm/"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url, head=False):
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "40"]
    if head:
        command.append("--head")
    result = subprocess.run(command + [url], capture_output=True, check=True)
    return result.stdout


class Elements(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def main():
    index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
    freeze_path = "docs/course-revision-20261016/freeze-01/sources/index.md"
    frozen = (ROOT / freeze_path).read_bytes()
    config = tomllib.loads((ROOT / "zensical.toml").read_text())
    headings = []
    for path in sorted((ROOT / "course/chapters").glob("*.md")):
        headings.extend(re.findall(r"^## ([A-Z\d]+\.\d+) (.+)$", path.read_text(), re.M))

    def nav_paths(entries):
        for entry in entries:
            if isinstance(entry, str):
                yield entry
            else:
                for value in entry.values():
                    yield from nav_paths(value) if isinstance(value, list) else [value]

    nav = list(nav_paths(config["project"]["nav"]))
    local = []
    for item in index:
        notebook = json.loads((ROOT / item["notebook"]).read_bytes())
        footer = export_course.section_footer(item, notebook, "main", True)
        tags = Elements(footer).tags
        local.append({
            "id": item["id"],
            "notebook": item["notebook"],
            "notebook_sha256": digest((ROOT / item["notebook"]).read_bytes()),
            "listed_in_navigation": nav.count(item["id"] + ".md") == 1,
            "footer_download": any(t == "a" and d.get("href") == item["notebook"] and "download" in d for t, d in tags),
            "footer_colab": any(t == "a" and d.get("href") == f"https://colab.research.google.com/github/{export_course.REPOSITORY}/blob/main/{item['notebook']}" for t, d in tags),
            "teaching_code_cells": sum(c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {}) for c in notebook["cells"]),
        })

    source_paths = [
        "scripts/export_course.py", "scripts/build_course.py", "course/lesson-index.json",
        "zensical.toml", "course/README.md", "course/first-steps.md", "course/chapters/01.md",
        "course/chapters/10.md", "course/chapters/12.md", "tiny_perceptron/multimodal.py",
        ".github/workflows/pages.yml",
    ]
    source_receipts = [{"path": p, "sha256": digest((ROOT / p).read_bytes())} for p in source_paths]
    extracts = {}
    for path in ["course/chapters/01.md", "course/chapters/10.md", "course/chapters/12.md"]:
        extracts[path] = (ROOT / path).read_text().split("\n## ")[0]
    guide = (ROOT / "course/README.md").read_text()
    extracts["guide_routes"] = [line for line in guide.splitlines() if any(word in line for word in ["圖片與聲音怎麼成為輸入", "如何教文風", "Dense、MoE", "回讀連結"]) ]
    extracts["warmup_headings"] = re.findall(r"^## (.+)$", (ROOT / "course/first-steps.md").read_text(), re.M)
    extracts["opening_backlinks"] = re.findall(r"\[[^\]]+\]\([^)]*first-steps\.md#[^)]+\)", (ROOT / "course/chapters/01.md").read_text())[:10]

    home = fetch(BASE).decode()
    info = json.loads(fetch(BASE + "build-info.json"))
    elements = Elements(home)
    worker_path = json.loads(re.search(r'<script id="__config"[^>]*>(.*?)</script>', home, re.S)[1])["search"]
    worker_url = urljoin(BASE, worker_path)
    worker = fetch(worker_url).decode()
    search_url = BASE + "search.json"
    search_raw = fetch(search_url)
    search = json.loads(search_raw)
    css_path = next(d["href"] for t, d in elements.tags if t == "link" and "main." in d.get("href", "") and "stylesheets" in d.get("href", ""))
    css_url = urljoin(BASE, css_path)
    css = fetch(css_url).decode()
    label = re.search(r'<label\b(?=[^>]*class="[^"]*md-header__button)(?=[^>]*for="__drawer")[^>]*>.*?</label>', home, re.S)
    sidebar = css.find("[data-md-toggle=drawer]:checked~.md-container .md-sidebar--primary")
    start = css.rfind("@media", 0, sidebar)
    search_fields = worker.find('fields:[C("title"')

    node_program = r'''
const vm = require("node:vm");
const fs = require("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const messages = [];
const context = { self: {postMessage: x => messages.push(x)}, console, Intl, TextEncoder, TextDecoder, setTimeout, clearTimeout };
vm.runInNewContext(input.worker, context, {timeout: 20000});
(async () => {
  await context.self.onmessage({data:{type:0,data:input.index}});
  const result = [];
  for (const term of input.terms) {
    await context.self.onmessage({data:{type:2,data:{input:term,filter:{input:{type:"operator",data:{operator:"and",operands:[]}},aggregation:{input:[{type:"term",data:{field:"tags"}}]}}}}});
    const message = messages.at(-1);
    result.push({term,type:message.type,total:message.data.total,pagination:message.data.pagination,items:message.data.items.slice(0,5).map(x=>({id:x.id,matches:x.matches.map(m=>({field:m.field})),document:input.index.items[x.id]}))});
  }
  console.log(JSON.stringify(result));
})().catch(e => {console.error(e);process.exitCode=1});
'''
    node = subprocess.run(["node", "-e", node_program], input=json.dumps({"worker": worker, "index": search, "terms": ["量化", "Unicode"]}), capture_output=True, text=True)
    search_result = {"exit_code": node.returncode, "stderr": node.stderr}
    if node.returncode == 0:
        results = json.loads(node.stdout)
        for result in results:
            for item in result["items"]:
                doc = item["document"]
                item["document"] = {"location": doc.get("location"), "title": doc.get("title"), "text_excerpt": re.sub("<[^>]+>", " ", doc.get("text", ""))[:220]}
        search_result["results"] = results
    else:
        search_result["stdout"] = node.stdout

    def inspect_lesson(item):
        url = BASE + item["id"] + ".html"
        raw = fetch(url)
        text = raw.decode()
        body = re.search(r"<article\b.*?</article>", text, re.S)[0]
        tags = Elements(body).tags
        downloads = [d["href"] for t, d in tags if t == "a" and "download" in d]
        colabs = [d["href"] for t, d in tags if t == "a" and d.get("href", "").startswith("https://colab.research.google.com/github/")]
        download_url = urljoin(url, downloads[0])
        headers = fetch(download_url, head=True).decode()
        code_blocks = sum("language-python" in d.get("class", "").split() for t, d in tags if t in {"div", "code"})
        output_blocks = body.count('class="output"')
        return {"id": item["id"], "url": url, "sha256": digest(raw), "code_blocks": code_blocks, "output_blocks": output_blocks, "download_href": downloads, "download_http_status": re.findall(r"HTTP/\S+ (\d+)", headers), "colab_href": colabs}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        live = list(pool.map(inspect_lesson, index))

    first_raw = fetch(BASE + "1.1.html").decode()
    first_output = html.unescape(re.search(r'<div class="output">.*?<pre>(.*?)</pre>', first_raw, re.S)[1])
    code = next(c for c in json.loads((ROOT / index[0]["notebook"]).read_bytes())["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {}))
    fresh = subprocess.run([sys.executable, "-c", export_course.joined(code["source"])], capture_output=True, text=True, check=True)
    print(json.dumps({
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-index/audit_index.py",
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "os": platform.platform(), "torch": importlib.metadata.version("torch"), "node": subprocess.check_output(["node", "--version"], text=True).strip(), "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0], "device": "CPU；網站資源以 HTTPS 讀取；未啟動訓練或雲端寫入"},
        "source_receipts": source_receipts,
        "frozen_home": {"path": freeze_path, "sha256": digest(frozen), "export_with_outputs_matches_exactly": export_course.home_introduction(index, True).encode() == frozen, "without_outputs_claims_results_present": "已有實際結果" in export_course.home_introduction(index, False)},
        "inventory": {"index_count": len(index), "unique_ids": len({i["id"] for i in index}), "chapter_heading_count": len(headings), "ids_and_titles_match_all_chapter_headings": {(i["id"], i["title"]) for i in index} == set(headings), "navigation_entries": len(nav), "all_lessons_listed_once": all(x["listed_in_navigation"] for x in local), "all_notebooks_present": len(local) == len(index), "all_generated_footers_have_download_and_colab": all(x["footer_download"] and x["footer_colab"] for x in local)},
        "curriculum_excerpts": extracts,
        "live_home": {"url": BASE, "sha256": digest(home.encode()), "build_info": info, "search_markup": re.search(r'<div class="md-search".*?</div>', home, re.S)[0], "drawer_header_label": label[0] if label else None, "drawer_input": [d for t, d in elements.tags if t == "input" and d.get("id") == "__drawer"], "mobile_css_url": css_url, "mobile_css_sha256": digest(css.encode()), "mobile_css_excerpt": css[start:sidebar + 380]},
        "search": {"index_url": search_url, "index_sha256": digest(search_raw), "index_items": len(search["items"]), "worker_url": worker_url, "worker_sha256": digest(worker.encode()), "worker_field_excerpt": worker[search_fields:search_fields + 160], "worker_execution": search_result},
        "live_lesson_summary": {"checked_pages": len(live), "all_pages_have_download_and_colab": all(x["download_href"] and x["colab_href"] for x in live), "all_notebook_downloads_http_200": all(x["download_http_status"][-1] == "200" for x in live), "pages_with_python_code": sum(x["code_blocks"] > 0 for x in live), "pages_with_python_but_no_output": [x["id"] for x in live if x["code_blocks"] > 0 and x["output_blocks"] == 0], "python_code_blocks": sum(x["code_blocks"] for x in live), "output_blocks": sum(x["output_blocks"] for x in live)},
        "fresh_opening_example": {"notebook": index[0]["notebook"], "command": [sys.executable, "-c", export_course.joined(code["source"])], "stdout": fresh.stdout, "live_first_output": first_output, "exact_match": fresh.stdout == first_output},
        "local_lessons": local,
        "live_lessons": live,
        "limitations": ["未重跑全部教材或長時間訓練；輸出來源另以發布流程與 checked_notebook 的檢查核對。", "本機沒有可用瀏覽器；手機抽屜依已發布的 HTML 關聯與媒體查詢 CSS 核對，未做手機視覺或點擊實測。", "Colab 僅核對入口與 Notebook 資源；未登入 Google 或執行雲端 Notebook。", "搜尋以發布的 worker 程式與完整 search.json 在 Node VM 執行；未做瀏覽器互動測試。"]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
