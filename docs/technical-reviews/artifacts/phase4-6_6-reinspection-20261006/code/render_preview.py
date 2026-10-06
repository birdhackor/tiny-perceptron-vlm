"""Render only the assigned local section preview at desktop and mobile sizes."""
import hashlib
import json
import platform
from pathlib import Path

from playwright.sync_api import sync_playwright

ART = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8765/6.6.html"
out = {"url": URL, "date": "2026-10-06", "platform": platform.platform(), "browser_executable": "/usr/bin/chromium", "renders": []}
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    out["browser_version"] = browser.version
    for name, width, height in (("desktop", 1280, 800), ("mobile", 390, 844)):
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        response = page.goto(URL, wait_until="networkidle", timeout=15000)
        assert response and response.status == 200
        page.locator("details").evaluate_all("els => els.forEach(e => e.open = true)")
        text = page.locator("body").inner_text()
        assert "邊界標記怎麼保持完整" in text and "BPE可登記整串特殊項" in text
        assert "<assistant>" in text and "<user>這只是引用文字" in text
        png = ART / "visual" / f"{name}.png"
        page.screenshot(path=str(png), full_page=True)
        html = page.content().encode("utf-8")
        (ART / "visual" / f"{name}.html").write_bytes(html)
        (ART / "visual" / f"{name}.txt").write_text(text, encoding="utf-8")
        extent = page.evaluate("({viewport:innerWidth,body:document.body.scrollWidth,document:document.documentElement.scrollWidth})")
        out["renders"].append({"name": name, "viewport": {"width": width, "height": height}, "http_status": response.status, "details_opened": True, "screenshot": str(png), "screenshot_sha256": hashlib.sha256(png.read_bytes()).hexdigest(), "rendered_html_sha256": hashlib.sha256(html).hexdigest(), "page_widths": extent})
        page.close()
    browser.close()
(ART / "execution/render-facts.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
