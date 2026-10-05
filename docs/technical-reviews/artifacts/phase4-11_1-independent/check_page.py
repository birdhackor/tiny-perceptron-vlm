"""Confirm the existing single-section page matches current prose/fences, then render it."""
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
body = (HERE / "original-fences/section.md").read_text()
url = "http://127.0.0.1:8765/11.1.html"
raw = urllib.request.urlopen(url, timeout=10).read()
(HERE / "page.html").write_bytes(raw)
soup = BeautifulSoup(raw, "html.parser")
article = soup.find("article")
prose = re.sub(r"```[^\n]*\n.*?```", "", body, flags=re.S)
paragraphs = []
for block in prose.split("\n\n"):
    block = block.strip()
    if not block or block.startswith(("##", "<")):
        continue
    block = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", block).replace("`", "")
    paragraphs.append(block)
page_text = article.get_text("", strip=False)
for paragraph in paragraphs:
    assert paragraph in page_text, paragraph
codes = [tag.get_text() for tag in article.select("pre code")]
for n in (1, 2):
    expected = (HERE / f"original-fences/fence-{n}.py").read_text().rstrip("\n")
    assert expected in [code.rstrip("\n") for code in codes]
title = body.splitlines()[0].removeprefix("## ")
assert title in article.get_text()
assert not article.select("img"), "11.1 has no referenced figure or embedded image"
viewports = []
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                                         args=["--no-sandbox", "--disable-gpu"])
    for label, width, height in (("desktop", 1280, 800), ("mobile", 390, 844)):
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.goto(url, wait_until="networkidle")
        page.screenshot(path=str(HERE / f"page-{label}.png"), full_page=True)
        viewports.append({"label": label, "width": width, "height": height,
                          "article_width": page.locator("article").bounding_box()["width"]})
        page.close()
    version = browser.version
    browser.close()
result = {"url": url, "section_sha256": hashlib.sha256(body.encode()).hexdigest(),
          "page_sha256": hashlib.sha256(raw).hexdigest(),
          "prose_blocks_matching_current_source": len(paragraphs), "python_fences_matching": 2,
          "referenced_figures": 0, "chromium": version, "viewports": viewports,
          "scope": "Existing 8765 single-section page; current section heading, all non-HTML prose blocks, both code fences match. No whole-book export. Screenshots are viewed separately; chapter introduction was read from raw Markdown."}
(HERE / "page-receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
