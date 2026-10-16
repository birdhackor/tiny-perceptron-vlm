"""Render the unedited 7.4 SVG with the installed Chromium; no file:// URL."""

import hashlib
import json
import platform
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[4]
OUT = Path("/workspace/work/textbook-repair-20261017/technical-7-4")
svg_path = ROOT / "course/figures/rewrite-07-04-answer-alignment.svg"
svg = svg_path.read_bytes()
records = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    for name, width, height in [("desktop", 640, 750), ("mobile", 390, 800)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.set_content('<!doctype html><meta charset="utf-8"><style>body{margin:0}svg{display:block;max-width:100%;height:auto}</style>' + svg.decode(), wait_until="load")
        page.evaluate("document.fonts.ready")
        destination = OUT / f"answer-alignment-{name}.png"
        page.screenshot(path=str(destination), full_page=True)
        records.append({"viewport": {"width": width, "height": height}, "screenshot": str(destination),
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
        page.close()
    version = browser.version
    browser.close()

print(json.dumps({
    "command": "uv run --frozen --extra cpu --no-sync python docs/technical-reviews/artifacts/repair-20261017-7-4/render_figure.py",
    "python": platform.python_version(), "chromium": version,
    "source": "course/figures/rewrite-07-04-answer-alignment.svg",
    "source_sha256": hashlib.sha256(svg).hexdigest(),
    "method": "page.set_content with raw unedited SVG bytes decoded as UTF-8; wrapper CSS only scales outer SVG; page.screenshot(full_page=True), no file URL, no browser download.",
    "screenshots": records,
    "scope": "Technical inspection of the SVG rendered at 640px and 390px; this is not a course website layout test."
}, ensure_ascii=False, indent=2))
