from pathlib import Path
import json
import markdown
from playwright.sync_api import sync_playwright

A = Path(__file__).resolve().parent
body = markdown.markdown((A / "section.md").read_text(), extensions=["tables", "fenced_code"])
html = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><style>body{font:18px/1.65 "Noto Sans CJK TC",sans-serif;margin:auto;max-width:900px;padding:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #999;padding:6px;text-align:center}pre{overflow:auto;background:#f3f4f6;padding:14px;font:14px/1.6 monospace}details{margin-top:24px}code{word-break:break-word}</style>' + body + '</html>'
(A / "section-render.html").write_text(html)
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"], timeout=15000)
    for width, height, name in [(1280, 800, "desktop"), (390, 844, "mobile")]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.set_content(html, timeout=15000)
        page.locator("details").evaluate_all("nodes => nodes.forEach(node => node.open = true)")
        page.screenshot(path=str(A / (name + ".png")), full_page=True, timeout=15000)
        print(json.dumps({"viewport": [width, height], "screenshot": name + ".png", "body_scroll_width": page.evaluate("document.body.scrollWidth"), "inner_width": page.evaluate("window.innerWidth"), "figures": page.locator("img").count()}, ensure_ascii=False))
        page.close()
    browser.close()
