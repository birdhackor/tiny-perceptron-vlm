"""Render the current section and necessary prerequisite SVGs without modification."""
from pathlib import Path
import hashlib
import json
import platform
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
figures = [
    ("natural_photo_evidence", 720, 1140),
    ("natural-v4-training-cat", 760, 646),
    ("practical_order", 720, 1120),
]
records = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                                args=["--no-sandbox"])
    version = browser.version
    for name, width, height in figures:
        source = ROOT / "course/figures" / (name + ".svg")
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=1)
        page.goto(source.as_uri())
        page.evaluate("document.fonts.ready")
        target = OUT / (name + ".png")
        page.screenshot(path=str(target), full_page=True)
        records.append({"source": str(source.relative_to(ROOT)),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "render": str(target.relative_to(ROOT)),
                        "render_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "viewport": [width, height]})
        page.close()
    browser.close()
print(json.dumps({"python": platform.python_version(), "chromium": version,
                  "records": records}, ensure_ascii=False, indent=2))
