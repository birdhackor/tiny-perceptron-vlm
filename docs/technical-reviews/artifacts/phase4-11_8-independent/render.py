"""Render and inspect only the existing 11.8 page; abort unrelated external resources."""
import hashlib
import json
import re
import urllib.request
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

BASE = Path(__file__).parent
ROOT = BASE.parents[3]
URL = "http://127.0.0.1:8765/11.8.html"
visual = BASE / "visual"
raw = urllib.request.urlopen(URL,timeout=15).read()
(visual / "served-11.8.html").write_bytes(raw)
source = (BASE / "original-fence/section.md").read_text()
expected = BeautifulSoup(markdown.markdown(source.replace("<details>", '<details markdown="1">'),extensions=["fenced_code","md_in_html"]), "html.parser")
norm = lambda s: re.sub(r"\s+", "", s)
expected_paragraphs = [p.get_text() for p in expected.find_all("p") if p.get_text().strip()]
inspection = {"url": URL, "served_html_sha256": hashlib.sha256(raw).hexdigest(), "external_resources": "Aborted HTTPS font/CDN resources so rendering is local and bounded; local styles/scripts/images were loaded", "desktop":{}, "mobile":{}}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium",headless=True,args=["--no-sandbox","--disable-gpu","--disable-dev-shm-usage","--disable-background-networking"])
    inspection["chromium_version"] = browser.version
    for name,w,h in [("desktop",1280,800),("mobile",390,844)]:
        page = browser.new_page(viewport={"width":w,"height":h},device_scale_factor=1)
        page.route("**/*",lambda route: route.continue_() if route.request.url.startswith("http://127.0.0.1:8765/") else route.abort())
        page.goto(URL,wait_until="domcontentloaded",timeout=15000)
        page.locator("article.md-content__inner").wait_for()
        page.wait_for_timeout(700)
        article = page.locator("article.md-content__inner")
        body = article.text_content()
        (visual / f"{name}-article-text.txt").write_text(body)
        (visual / f"{name}-paragraph-match.json").write_text(json.dumps([{"source":t,"matched":norm(t) in norm(body)} for t in expected_paragraphs],ensure_ascii=False,indent=2)+"\n")
        assert all(norm(t) in norm(body) for t in expected_paragraphs)
        code = article.locator("pre code").first.inner_text()
        assert norm(code) == norm((BASE / "original-fence/fence-1.py").read_text())
        imgs = article.locator('img[src*="rewrite-11-shape-pairs.svg"]')
        assert imgs.count() == 1
        img = imgs.first
        img.scroll_into_view_if_needed()
        assert img.evaluate("e=>e.complete && e.naturalWidth===640 && e.naturalHeight===650")
        source_url = img.evaluate("e=>e.src")
        image_bytes = urllib.request.urlopen(source_url,timeout=15).read()
        assert hashlib.sha256(image_bytes).hexdigest() == hashlib.sha256((ROOT / "course/figures/rewrite-11-shape-pairs.svg").read_bytes()).hexdigest()
        page.screenshot(path=str(visual / f"{name}-figure-viewport.png"))
        article.screenshot(path=str(visual / f"{name}-article.png"),timeout=15000)
        page.evaluate("window.scrollTo(0,0)")
        page.wait_for_timeout(200)
        page.screenshot(path=str(visual / f"{name}-page.png"),full_page=True,timeout=15000)
        inspection[name] = {"viewport":{"width":w,"height":h}, "all_source_prose_paragraphs_matched":len(expected_paragraphs), "original_fence_matched":True, "image_url":source_url, "image_sha256":hashlib.sha256(image_bytes).hexdigest(), "image_box":img.bounding_box(), "body_text":body}
        page.close()
    browser.close()
(visual / "render-receipt.json").write_text(json.dumps(inspection,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"chromium":inspection["chromium_version"],"source_paragraphs_matched":len(expected_paragraphs),"desktop_mobile_rendered":True,"image_sha256":inspection["desktop"]["image_sha256"]},ensure_ascii=False))
