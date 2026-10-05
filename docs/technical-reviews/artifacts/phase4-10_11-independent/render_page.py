import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
url = "http://127.0.0.1:8765/10.11.html"
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium",
        args=["--no-sandbox", "--disable-dev-shm-usage"])
    receipts = []
    for name, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        response = page.goto(url, wait_until="networkidle")
        displayed_code = page.locator("main pre code").inner_text()
        expected_code = (OUT / "fence-1.py").read_text()
        assert displayed_code.strip() == expected_code.strip()
        page.screenshot(path=str(OUT / f"page-{name}.png"), full_page=True)
        main = page.locator("main").inner_text()
        (OUT / f"page-{name}-main.txt").write_text(main)
        receipts.append({"url": url, "status": response.status, "viewport": [width, height],
            "main_sha256": hashlib.sha256(main.encode()).hexdigest(),
            "body_scroll_width": page.evaluate("document.body.scrollWidth"),
            "viewport_width": width, "images": page.locator("main img").count(),
            "heading": page.locator("main h1").inner_text() if page.locator("main h1").count() else None,
            "displayed_code_matches_original": True,
            "code_scroll": page.locator("main pre code").evaluate("e => ({client: e.clientWidth, scroll: e.scrollWidth, overflowX: getComputedStyle(e).overflowX})"),
            "chromium_version": browser.version})
        if name == "mobile":
            page.locator("main pre code").evaluate("e => e.scrollLeft = e.scrollWidth")
            page.locator("main pre").screenshot(path=str(OUT / "page-mobile-code-right.png"))
        page.close()
    browser.close()
(OUT / "page-render-receipts.json").write_text(json.dumps(receipts, indent=2, ensure_ascii=False)+"\n")
print(json.dumps(receipts, ensure_ascii=False))
