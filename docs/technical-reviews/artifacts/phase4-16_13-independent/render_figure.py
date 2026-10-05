"""Render the frozen SVG directly in Chromium without network or file navigation."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 640, "height": 1175}, device_scale_factor=1)
    page.set_content('<html><head><style>body{margin:0}</style></head><body>' +
                     (HERE / "inputs/visible-pairs.svg").read_text() + '</body></html>')
    page.screenshot(path=str(HERE / "figure-render.png"), full_page=True)
    environment = {"browser": browser.version, "executable_path": "/usr/bin/chromium",
                   "viewport": "640x1175", "device_scale_factor": "1", "svg_count": str(page.locator('svg').count()),
                   "font_query": str(page.evaluate("document.fonts.check('28px \"Noto Sans CJK TC\"')"))}
    (HERE / "figure-render-env.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(environment))
    browser.close()
