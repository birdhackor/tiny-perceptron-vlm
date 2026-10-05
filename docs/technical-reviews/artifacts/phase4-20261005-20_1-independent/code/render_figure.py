"""Render only the reviewed frozen SVG at desktop and mobile display widths."""
from pathlib import Path
import hashlib
import json
from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parents[1]
figure = BASE / "figures/rewrite-20-input-routes.svg"
records = []
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    for name, width in [("desktop", 640), ("mobile", 366)]:
        page = browser.new_page(viewport={"width": width, "height": round(669 * width / 640)}, device_scale_factor=1)
        page.set_content('<html><head><style>html,body{margin:0}svg{display:block;width:100%;height:auto}</style></head><body>' + figure.read_text() + '</body></html>')
        page.evaluate("document.fonts.ready")
        path = BASE / "figures" / (name + ".png")
        page.screenshot(path=str(path), full_page=True)
        records.append({"path": str(path.relative_to(BASE)), "width": width, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        page.close()
    print(json.dumps({"renderer": browser.version, "source_sha256": hashlib.sha256(figure.read_bytes()).hexdigest(), "renders": records}))
    browser.close()
