"""Inspect the supplied existing local preview; no site/server mutation."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
records = []
with sync_playwright() as runner:
    browser = runner.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--disable-gpu"])
    for label, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.goto("http://127.0.0.1:8765/7.5.html", wait_until="domcontentloaded", timeout=20000)
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(HERE / f"preview-{label}.png"), full_page=True)
        code = page.locator("pre").filter(has_text="問題位置logits梯度").first
        code.screenshot(path=str(HERE / f"preview-{label}-current-code.png"))
        records.append({"viewport": {"width": width, "height": height}, "label": label,
            "current_code_text": code.inner_text(), "renderer": "Playwright Chromium", "browser_version": browser.version})
        page.close()
    browser.close()
(HERE / "preview-browser-receipt.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"viewports": len(records), "browser_version": records[0]["browser_version"]}, ensure_ascii=False))
