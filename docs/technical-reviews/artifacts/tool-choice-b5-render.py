"""Render the actual B.5 SVG for independent visual inspection."""

import importlib.metadata
import json
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ART = Path(__file__).resolve().parent
ROOT = ART.parents[2]
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path="/usr/bin/chromium")
    page = browser.new_page(viewport={"width": 760, "height": 570}, device_scale_factor=2)
    svg = (ROOT / "course/figures/tool_choice.svg").read_text()
    page.set_content('<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">' + svg + "</body></html>")
    page.screenshot(path=str(ART / "tool-choice-b5-figure-render.png"), full_page=True)
    record = {"python": sys.version, "playwright": importlib.metadata.version("playwright"), "chromium": browser.version, "viewport": "760x570", "scale": 2, "rendered_pixels": list(Image.open(ART / "tool-choice-b5-figure-render.png").size), "render_method": "page.set_content read-only local SVG, body margin 0 (full-page capture includes the inline SVG baseline margin)", "svg": "course/figures/tool_choice.svg", "render": "docs/technical-reviews/artifacts/tool-choice-b5-figure-render.png"}
    (ART / "tool-choice-b5-render-record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False))
    browser.close()
