"""Capture current section with bounded local-only browser requests."""
import hashlib
import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8765/12.5.html"
section = (OUT / "original-fence/section.md").read_text()
paragraphs = [x for x in section.split("\n\n") if x and not x.startswith(("##", "![", "```", "<", "可回顧"))]
fence = (OUT / "original-fence/fence-1.py").read_text()
normalize = lambda s: re.sub(r"\s+", "", s.replace("`", ""))
records = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
        args=["--no-sandbox", "--disable-gpu", "--disable-background-networking"])
    for name, size in [("desktop", {"width": 1280, "height": 800}),
                       ("mobile", {"width": 390, "height": 844})]:
        context = browser.new_context(viewport=size, device_scale_factor=1)
        context.route("**/*", lambda route: route.continue_() if route.request.url.startswith("http://127.0.0.1:8765/") else route.abort())
        page = context.new_page()
        response = page.goto(URL, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(700)
        article = page.locator("article")
        article_text = article.inner_text()
        checks = [normalize(x) in normalize(article_text) for x in paragraphs]
        code_check = normalize(fence) in normalize(article_text)
        assert all(checks) and code_check
        images = article.locator("img").evaluate_all("els => els.map(e => ({src:e.src, naturalWidth:e.naturalWidth, naturalHeight:e.naturalHeight, rect:{x:e.getBoundingClientRect().x,y:e.getBoundingClientRect().y,width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height}}))")
        for img in images:
            if "multimodal_audio_axes.svg" in img["src"]:
                raw = context.request.get(img["src"]).body()
                expected = (OUT / "inputs/course/figures/multimodal_audio_axes.svg").read_bytes()
                assert raw == expected and img["naturalWidth"] > 0
                img["sha256"] = hashlib.sha256(raw).hexdigest()
        page.screenshot(path=str(OUT / f"page-{name}.png"), full_page=True, timeout=15000)
        records.append({"viewport":size, "status":response.status, "prose_match":checks,
                        "fence_matches":code_check, "images":images,
                        "document_scroll_width":page.evaluate("document.documentElement.scrollWidth"),
                        "browser_version":browser.version})
        context.close()
    browser.close()
print(json.dumps({"url":URL, "network_scope":"local page/assets only", "records":records}, ensure_ascii=False, indent=2))
