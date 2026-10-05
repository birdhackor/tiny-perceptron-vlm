"""Render the actual frozen SVG at native, desktop, and mobile display sizes."""
import base64
import hashlib
import json
import platform
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

directory = Path(__file__).resolve().parent
svg = directory.parent / "inputs/rewrite-16-12-sliding-path.svg"
data = svg.read_bytes()
html = '<html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{margin:0;padding:16px;background:white}img{display:block;width:640px;max-width:100%;height:auto}</style></head><body><img alt="16.12 actual figure" src="data:image/svg+xml;base64,' + base64.b64encode(data).decode() + '"></body></html>'
records = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    version = browser.version
    for label, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.set_content(html, wait_until="load")
        page.locator("img").evaluate("el => el.decode()")
        page.screenshot(path=str(directory / (label + ".png")))
        if label == "desktop":
            page.locator("img").screenshot(path=str(directory / "native.png"))
        records.append({"viewport": {"width": width, "height": height}, "label": label, "image_box": page.locator("img").bounding_box()})
        page.close()
    browser.close()
print(json.dumps({"source_sha256": hashlib.sha256(data).hexdigest(), "browser": version, "python": sys.version, "platform": platform.platform(), "screenshots": records, "scope": "Actual SVG embedded in a minimal image page; whole chapter browser layout was not tested."}, ensure_ascii=False, indent=2))
