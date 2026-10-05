"""Render the current locally served lesson with bounded browser navigation."""
import json
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

base = Path(__file__).resolve().parent
records = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                               args=["--no-sandbox", "--disable-gpu", "--disable-background-networking"])
    for name, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        blocked = []

        def route(request):
            host = urlsplit(request.request.url).hostname
            if host in ["127.0.0.1", "localhost"] or request.request.url.startswith("data:"):
                request.continue_()
            else:
                blocked.append(request.request.url)
                request.abort()

        page.route("**/*", route)
        response = page.goto("http://127.0.0.1:8765/12.13.html", wait_until="domcontentloaded", timeout=20000)
        image = page.locator('article img[src$="/figures/rewrite-12-time-order.svg"]')
        image.wait_for(state="visible", timeout=10000)
        page.wait_for_function("Array.from(document.querySelectorAll('article img')).every(x => x.complete && x.naturalWidth > 0)", timeout=10000)
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(base / f"{name}.png"), full_page=True, timeout=15000)
        image.screenshot(path=str(base / f"{name}-figure.png"), timeout=15000)
        record = {"name": name, "viewport": [width, height], "http_status": response.status,
                  "browser_version": browser.version, "blocked_external_requests": blocked,
                  "figure_geometry": image.evaluate("x => ({naturalWidth:x.naturalWidth,naturalHeight:x.naturalHeight,width:x.getBoundingClientRect().width,height:x.getBoundingClientRect().height})")}
        records.append(record)
        page.close()
    browser.close()
(base / "render-record.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(records, ensure_ascii=False, indent=2))
