"""Bounded real Chromium render of the current section preview."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
url = "http://127.0.0.1:8765/7.1.html"
started = time.perf_counter()
records = []
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    for name, width, height in (("desktop", 1280, 900), ("mobile", 390, 844)):
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.set_default_timeout(15000)
        response = page.goto(url, wait_until="networkidle", timeout=20000)
        assert response and response.status == 200
        article = page.locator("article").first
        text = article.inner_text()
        assert "7.1" in text and "帶著舊錯答案" in text
        assert "[1,3,57,51,57,69,71,2,4,58]" in text.replace(" ", "")
        (HERE / f"render-{name}-article.txt").write_text(text, encoding="utf-8")
        page.screenshot(path=str(HERE / f"render-{name}.png"), full_page=True)
        records.append({"name": name, "viewport": [width, height], "http_status": response.status, "article_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "screenshot_sha256": hashlib.sha256((HERE / f"render-{name}.png").read_bytes()).hexdigest(), "current_orthography_present": True, "x_ids_present": True})
        page.close()
    version = browser.version
    browser.close()
receipt = {"command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-7_1-reinspection-20261006/render_page.py", "url": url, "python": sys.version, "platform": platform.platform(), "browser": "Chromium " + version, "device": "CPU headless renderer", "elapsed_seconds": time.perf_counter() - started, "pages": records, "scope": "Actual current-page presentation; screenshots must be personally viewed. Does not execute course model code."}
(HERE / "render-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, ensure_ascii=False, indent=2))
