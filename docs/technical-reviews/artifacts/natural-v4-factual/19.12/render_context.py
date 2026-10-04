"""Render the newly linked RMSNorm prerequisite SVG with existing Chromium."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[5]
source = root / "course/figures/normalization.svg"
target = Path(__file__).with_name("normalization-context.png")
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 800, "height": 330}, device_scale_factor=1)
    page.emulate_media(reduced_motion="reduce")
    page.set_content('<html><body style="margin:0">' + source.read_text() + '</body></html>', wait_until="load")
    page.screenshot(path=str(target))
    version = browser.version
    browser.close()
print(json.dumps({"source": source.relative_to(root).as_posix(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "render": target.relative_to(root).as_posix(), "render_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "renderer": "existing Chromium " + version, "viewport": "800x330", "reduced_motion": True}, indent=2))
