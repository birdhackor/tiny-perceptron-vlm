from pathlib import Path
import hashlib
import json
import platform
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "inputs/multimodal_overlapping_bands.svg"
TARGET = OUT / "context-mel-figure.png"
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1200, "height": 800}, device_scale_factor=1)
    page.set_content(SOURCE.read_text(encoding="utf-8"))
    page.locator("svg").screenshot(path=str(TARGET))
    version = browser.version
    browser.close()
metadata = {
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "render_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
    "method": "Chromium renders the exact original SVG as page content; SVG element screenshot",
    "chromium": version,
    "python": platform.python_version(),
    "viewport": "1200x800",
    "network": "No network request; set_content from exact local original bytes",
    "initial_attempt": {
        "command": ".venv/bin/python inline Playwright script; page.goto(SOURCE.resolve().as_uri())",
        "exit_code": 1,
        "error": "Page.goto: net::ERR_BLOCKED_BY_ADMINISTRATOR at file:///workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-g_3-factual-independent-20261005/inputs/multimodal_overlapping_bands.svg",
        "interpretation": "Browser policy blocks file navigation. No claim or source change; use the original SVG with page.set_content.",
    },
}
(OUT / "render.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(metadata, ensure_ascii=False))
