"""Render the frozen B.1 and its intro for actual visual inspection."""
import json
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

art = Path(__file__).resolve().parent
body = (art / "frozen-input/chapter-intro.md").read_text(encoding="utf-8") + (art / "frozen-input/B.1.md").read_text(encoding="utf-8")
html = markdown.markdown(body.replace("<details>", '<details open markdown="1">'), extensions=["fenced_code", "tables", "md_in_html"])
page_html = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>B.1 frozen factual input</title><style>body{font:18px/1.7 sans-serif;color:#202124;max-width:900px;margin:32px auto;padding:0 20px}h1{font-size:30px}h2{font-size:27px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f4f5;padding:14px;font-size:14px;line-height:1.6}code{overflow-wrap:anywhere}summary{font-weight:600}a{color:#245ec3}</style><body>' + html + '</body></html>'
(art / "rendered-B.1.html").write_text(page_html, encoding="utf-8")
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    inspected = []
    for label, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.set_content(page_html, wait_until="load")
        page.screenshot(path=str(art / f"B.1-{label}.png"), full_page=True)
        inspected.append({"viewport": label, "width": width, "height": height, "headings": page.locator("h1,h2").all_text_contents(), "image_count": page.locator("img").count(), "scroll_width": page.evaluate("document.documentElement.scrollWidth"), "browser_version": browser.version})
        page.close()
    browser.close()
(art / "render-result.json").write_text(json.dumps(inspected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(inspected, ensure_ascii=False))
