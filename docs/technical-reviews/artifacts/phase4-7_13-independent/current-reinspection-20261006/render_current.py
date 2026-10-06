"""Render the supplied current local page with bounded Chromium timeouts."""
import hashlib
import json
import platform
from pathlib import Path

from playwright.sync_api import sync_playwright

R = Path(__file__).resolve().parent
results = {"python": platform.python_version(), "url": "http://127.0.0.1:8765/7.13.html", "renders": []}
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                                args=["--no-sandbox", "--disable-dev-shm-usage"], timeout=15000)
    results["chromium"] = browser.version
    for name, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=1)
        page = context.new_page()
        page.set_default_timeout(10000)
        response = page.goto(results["url"], wait_until="domcontentloaded", timeout=15000)
        assert response.status == 200
        text = page.locator("body").inner_text()
        assert "改風格" in text and "answer欄位" in text
        page.screenshot(path=str(R / f"{name}-top.png"), timeout=10000)
        supplemental = page.locator("details").filter(has_text="補充：實作約定與原始紀錄").first
        supplemental.evaluate("element => { element.open = true; }")
        supplemental.scroll_into_view_if_needed()
        page.screenshot(path=str(R / f"{name}-supplement.png"), timeout=10000)
        results["renders"].append({"viewport": {"name": name, "width": width, "height": height},
                                   "status": response.status, "supplement_opened": True,
                                   "body_scroll_width": page.evaluate("document.body.scrollWidth"),
                                   "inner_width": page.evaluate("window.innerWidth"),
                                   "screenshots": [{"path": str(R / f"{name}-{part}.png"),
                                                    "sha256": hashlib.sha256((R / f"{name}-{part}.png").read_bytes()).hexdigest()}
                                                   for part in ["top", "supplement"]]})
        context.close()
    browser.close()
(R / "render-receipt.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2))
