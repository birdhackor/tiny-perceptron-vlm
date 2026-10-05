"""Bounded local Chromium rendering of the original SVG; no web service."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
svg = (ROOT / "inputs/course/figures/rewrite-08-new-8.14-selected-lines.svg").read_bytes()
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        executable_path="/usr/bin/chromium",
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--disable-background-networking"],
        timeout=10000,
    )
    try:
        context = browser.new_context(viewport={"width": 640, "height": 760}, device_scale_factor=1)
        context.route("**/*", lambda route: route.abort())
        page = context.new_page()
        page.set_content('<html><body style="margin:0">' + svg.decode("utf-8") + "</body></html>", wait_until="domcontentloaded", timeout=10000)
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(ROOT / "figure.png"), timeout=10000)
        receipt = {
            "method": "Playwright set_content of frozen original SVG, 640x760 native viewport; no file URL or service",
            "browser": browser.version,
            "playwright": importlib.metadata.version("playwright"),
            "python": sys.version,
            "platform": platform.platform(),
            "svg_sha256": hashlib.sha256(svg).hexdigest(),
            "screenshot_sha256": hashlib.sha256((ROOT / "figure.png").read_bytes()).hexdigest(),
            "device": "CPU; Chromium --disable-gpu",
            "network": "context routes aborted",
            "status": "rendered",
        }
        (ROOT / "playwright-render-results.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        context.close()
    finally:
        browser.close()
