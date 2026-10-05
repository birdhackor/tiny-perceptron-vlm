"""Render the already served 10.9 HTML and verify current original content before capture."""
import hashlib
import json
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

base = Path(__file__).resolve().parents[1]
body = (base / "original/section.md").read_bytes()
reference_html = markdown.markdown(body.decode(), extensions=["fenced_code"])
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    context = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
    # Keep rendering bounded to this internal preview and prevent external scripts/network dependencies.
    context.route("**/*", lambda route: route.continue_() if route.request.url.startswith("http://127.0.0.1:8765/") else route.abort())
    page = context.new_page()
    response = page.goto("http://127.0.0.1:8765/10.9.html", wait_until="networkidle")
    assert response.status == 200
    article = page.locator("article.md-content__inner")
    actual_blocks = article.locator("p,pre,h1,h2,summary").all_text_contents()
    reference = context.new_page()
    reference.set_content(reference_html)
    reference_blocks = reference.locator("p,pre,h1,h2,summary").all_text_contents()
    normalize = lambda x: "".join(x.split()).replace("¶", "")
    actual_normalized = [normalize(x) for x in actual_blocks]
    for block in reference_blocks:
        assert normalize(block) in actual_normalized, block
    assert article.locator("img").count() == 0
    (base / "page-article.html").write_text(article.inner_html(), encoding="utf-8")
    article.locator("details summary").click()
    screenshots = []
    for name, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page.set_viewport_size({"width": width, "height": height})
        target = base / (name + ".png")
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(target), full_page=True)
        screenshots.append({"viewport": [width,height], "path": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "article_box": article.bounding_box()})
    print(json.dumps({"url": page.url, "http_status": response.status, "source_sha256": hashlib.sha256(body).hexdigest(), "current_source_blocks_checked": len(reference_blocks), "figures": 0, "screenshots": screenshots, "browser": browser.version, "inspection_scope": "The 10.9 article only, including the original code and unfolded original recall/evidence links. Every source text block was found in the served article after whitespace and heading permalink normalization; no source text was inferred from a build result."}, indent=2, ensure_ascii=False))
    browser.close()
