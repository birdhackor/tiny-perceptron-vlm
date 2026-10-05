"""Render and screenshot the exact A.2 SVG without changing it."""
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
svg = (HERE / "inputs/rewrite-A-rag-flow.svg").read_text()
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
    page.set_content('<html><body style="margin:16px;background:white">' + svg + '</body></html>')
    page.locator("svg").screenshot(path=str(HERE / "rag-flow.png"))
    print({"renderer": "chromium", "browser_version": browser.version, "svg_bbox": page.locator("svg").bounding_box(), "output": "rag-flow.png"})
    browser.close()
