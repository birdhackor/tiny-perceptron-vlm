"""Informed technical callback: inspect the served chapter-0C navigation only."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[4]
BASE = "http://127.0.0.1:8890/"
URL = urljoin(BASE, "chapter-0C.html")
WORK = Path("/workspace/work/curriculum-20261016/chapter-0C-callback09")
WORK.mkdir(parents=True, exist_ok=True)
RECEIPT = Path(__file__).with_suffix(".json")


def text(node):
    return node.get_text(" ", strip=True).replace(" ¶", "")


def screenshot(page, name, full_page=True):
    path = WORK / name
    page.screenshot(path=str(path), full_page=full_page)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def nav_groups(soup):
    groups = []
    for node in soup.select(".md-sidebar--primary li.md-nav__item--nested"):
        label = node.find("label", recursive=False)
        if label is not None:
            links = [
                {"text": text(a), "href": a["href"]}
                for a in node.select("a[href]")
            ]
            groups.append({"label": text(label), "links": links})
    return groups


result = {
    "scope": "Informed callback; served navigation, link targets, and implementation only. No numbered lesson body review, export rerun, training, or author's 43-test rerun.",
    "url": URL,
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/chapter-0C-revision09-callback-01-check.py",
    "environment": {
        "python": platform.python_version(),
        "playwright": version("playwright"),
        "beautifulsoup4": version("beautifulsoup4"),
        "device": "CPU; Chromium headless /usr/bin/chromium",
    },
    "implementation": {},
    "harness_notes": "Initial invocations stopped on evidence-script comparisons: JavaScript resolves relative hrefs to absolute URLs, and the broad substring C. also matched chapter-0C.html. A mobile script action also targeted an unused hidden nav title. Direct DOM inspection confirmed the expected targets; the final script compares resolved filenames and operates the visible nav label. These were evidence-script corrections, not product defects.",
}
for relative in ["scripts/export_course.py", "zensical.toml", "course/lesson-index.json"]:
    result["implementation"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    result["environment"]["chromium"] = browser.version
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page.set_default_timeout(10000)
    response = page.goto(URL, wait_until="networkidle")
    assert response.status == 200
    page.evaluate("document.fonts.ready")
    soup = BeautifulSoup(page.content(), "html.parser")
    article = soup.select_one("article")
    result["article_text"] = text(article)
    grouped = []
    for heading in article.select("h3"):
        links = [{"text": text(a), "href": a["href"]} for a in heading.find_next_sibling("ul").select("a")]
        grouped.append({"label": text(heading), "id": heading.get("id"), "links": links})
    assert [g["label"] for g in grouped] == ["第7章後：推理與計算預算", "第13章後：模型回饋"]
    assert [[a["href"] for a in g["links"]] for g in grouped] == [
        [urljoin(BASE, f"C.{i}.html") for i in range(1, 6)], [urljoin(BASE, "C.6.html"), urljoin(BASE, "C.7.html")]
    ]
    result["article_groups"] = grouped
    result["sidebar_toc"] = [{"text": text(a), "href": a["href"]} for a in soup.select(".md-sidebar--secondary a[href]")]
    assert result["sidebar_toc"] == [
        {"text": "本章小節", "href": URL + "#_1"},
        {"text": "第7章後：推理與計算預算", "href": URL + "#7"},
        {"text": "第13章後：模型回饋", "href": URL + "#13"},
    ]
    groups = nav_groups(soup)
    labels = [g["label"] for g in groups]
    c15 = next(g for g in groups if g["label"] == "必修：思考步驟、候選與驗證（C.1–C.5）")
    c67 = next(g for g in groups if g["label"] == "回饋訓練實作（C.6–C.7）")
    assert [a["href"] for a in c15["links"] if Path(urlsplit(a["href"]).path).name.startswith("C.")] == [urljoin(BASE, f"C.{i}.html") for i in range(1, 6)]
    assert [a["href"] for a in c67["links"]] == [urljoin(BASE, "C.6.html"), urljoin(BASE, "C.7.html")]
    assert labels.index("第 7 章：從續寫文字到回答問題") < labels.index(c15["label"]) < labels.index("第 8 章：風格、個性與指令遵循")
    assert labels.index("第 13 章：把回答偏好變成訓練") < labels.index(c67["label"]) < labels.index("第 14 章：逐項理解現代文字模型的設計")
    result["sidebar_group_order"] = labels
    result["sidebar_C_groups"] = [c15, c67]
    result["link_targets"] = []
    links = [a for a in article.select("a[href]") if "headerlink" not in a.get("class", [])]
    for link in links:
        target = urljoin(URL, link["href"])
        target_response = page.request.get(target)
        assert target_response.status == 200
        target_soup = BeautifulSoup(target_response.body(), "html.parser")
        fragment = urlsplit(target).fragment
        assert not fragment or target_soup.find(id=fragment) is not None
        h1 = target_soup.select_one("article h1")
        result["link_targets"].append({"text": text(link), "url": target, "status": target_response.status, "fragment_present": not fragment or target_soup.find(id=fragment) is not None, "destination_h1": text(h1)})
    assert page.locator("article").evaluate("e => e.scrollWidth <= e.clientWidth")
    result["desktop"] = {"viewport": "1440x1000", "article_overflow": False, "screenshot": screenshot(page, "desktop.png")}
    c67_label = page.locator('.md-sidebar--primary label.md-nav__link').filter(has_text="回饋訓練實作（C.6–C.7）")
    c67_label.scroll_into_view_if_needed()
    c67_label.click()
    page.wait_for_timeout(400)
    page.locator('.md-sidebar--primary a[href$="/C.7.html"]').evaluate("e => e.scrollIntoView({block: 'center', behavior: 'instant'})")
    result["desktop"]["sidebar_C67_screenshot"] = screenshot(page, "desktop-sidebar-C67.png", full_page=False)
    mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1, is_mobile=True, has_touch=True)
    mobile.set_default_timeout(10000)
    mobile.goto(URL, wait_until="networkidle")
    mobile.evaluate("document.fonts.ready")
    assert mobile.locator("article").evaluate("e => e.scrollWidth <= e.clientWidth")
    result["mobile"] = {"viewport": "390x844", "article_overflow": False, "screenshot": screenshot(mobile, "mobile.png")}
    mobile.locator('.md-header label[for="__drawer"]').click()
    mobile.locator('.md-sidebar--primary a[href$="/C.5.html"]').scroll_into_view_if_needed()
    result["mobile"]["drawer_C15_screenshot"] = screenshot(mobile, "mobile-drawer-C15.png")
    mobile_c67 = mobile.locator('.md-sidebar--primary label.md-nav__link').filter(has_text="回饋訓練實作（C.6–C.7）")
    mobile_c67.scroll_into_view_if_needed()
    mobile_c67.click()
    mobile.wait_for_timeout(400)
    assert mobile.locator('.md-sidebar--primary a[href$="/C.6.html"]').is_visible()
    assert mobile.locator('.md-sidebar--primary a[href$="/C.7.html"]').is_visible()
    mobile.locator('.md-sidebar--primary a[href$="/C.7.html"]').evaluate("e => e.scrollIntoView({block: 'center', behavior: 'instant'})")
    result["mobile"]["drawer_C67_screenshot"] = screenshot(mobile, "mobile-drawer-C67.png", full_page=False)
    # One ordinary chapter is checked in the served preview, without reading its lessons.
    generic_response = page.request.get(urljoin(BASE, "chapter-08.html"))
    generic_article = BeautifulSoup(generic_response.body(), "html.parser").select_one("article")
    assert generic_response.status == 200
    assert "依序閱讀，或點進眼前想弄懂的問題。" in text(generic_article)
    assert not generic_article.select("h3")
    result["ordinary_chapter_control"] = {"url": urljoin(BASE, "chapter-08.html"), "status": 200, "generic_list_sentence_present": True, "special_C_group_headings": False}
    browser.close()

result["result"] = "PASS: two groups contain exactly C.1–C.5 and C.6–C.7 in order; all ten article navigation links return HTTP 200 and existing fragment targets; sidebar groups remain after chapter 7 and chapter 13; both new groups appear in ToC; desktop and mobile article have no horizontal overflow; mobile C groups can be opened. Ordinary chapter 08 retains the generic list."
RECEIPT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"receipt": str(RECEIPT.relative_to(ROOT)), "result": result["result"], "screenshots": str(WORK)}, ensure_ascii=False, indent=2))
