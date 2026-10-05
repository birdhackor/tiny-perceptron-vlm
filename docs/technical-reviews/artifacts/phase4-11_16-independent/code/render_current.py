"""Verify the current isolated lesson page and render it at two viewports."""
from pathlib import Path
import hashlib
import json
import sys
import markdown
import re

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import importlib.util
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

spec = importlib.util.spec_from_file_location("section_facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
section, whole, first = facts.original_section(ROOT / "course/chapters/11.md", "11.16")
fences = facts.fences(section, first)
out = Path(__file__).resolve().parents[1] / "visual"
response = requests.get("http://127.0.0.1:8765/11.16.html", timeout=15)
response.raise_for_status()
soup = BeautifulSoup(response.content, "html.parser")
assert [c.get_text() for c in soup.select("pre code")] == [f["raw"].decode() for f in fences]
headings = soup.select("h1,h2")
assert len(headings) == 1 and "11.16" in headings[0].get_text()
comparison_md = re.sub(r"(?m)^</?details>\s*$|^<summary>.*?</summary>\s*$", "", section.decode())
expected_soup = BeautifulSoup(markdown.markdown(comparison_md, extensions=["fenced_code"]), "html.parser")
expected_prose = [p.get_text() for p in expected_soup.select("p") if not p.select("img")]
served_prose = [p.get_text() for p in soup.select("p") if not p.select("img")]
assert served_prose[:len(expected_prose)] == expected_prose
page_additions = served_prose[len(expected_prose):]
assert soup.select_one("img")["alt"] in section.decode()
figure = requests.get("http://127.0.0.1:8765/figures/rewrite-11-line-order.svg", timeout=15)
figure.raise_for_status()
assert figure.content == (ROOT / "course/figures/rewrite-11-line-order.svg").read_bytes()
(out / "current-page.html").write_bytes(response.content)
(out / "served-figure.svg").write_bytes(figure.content)
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    records = []
    for name, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.goto("http://127.0.0.1:8765/11.16.html", wait_until="networkidle")
        page.locator("summary").click()
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(out / f"{name}.png"), full_page=True, animations="disabled")
        records.append({"name": name, "viewport": {"width": width, "height": height}, "image_complete": page.locator("img").evaluate("e => e.complete && e.naturalWidth > 0"), "image_box": page.locator("img").bounding_box(), "page_dimensions": page.evaluate("({width:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight})")})
        page.close()
    manifest = {"url": "http://127.0.0.1:8765/11.16.html", "browser_version": browser.version, "section_sha256": hashlib.sha256(section).hexdigest(), "served_figure_matches_current_bytes": True, "fences_match_current_bytes": True, "all_lesson_prose_paragraphs_match_current_text": True, "extra_builder_paragraphs": page_additions, "renders": records}
    browser.close()
(out / "render-provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
