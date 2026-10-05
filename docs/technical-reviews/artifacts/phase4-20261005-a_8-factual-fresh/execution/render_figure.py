"""Render the unchanged source SVG in system Chromium at desktop/mobile widths."""
import hashlib
import json
import platform
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

ART = Path(__file__).resolve().parent.parent
svg = ART / "inputs/course/figures/rewrite-A-clue-position.svg"
text = svg.read_text()
measurements = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True,
                                args=["--no-sandbox", "--disable-dev-shm-usage"])
    for width in [850, 640, 390]:
        page = browser.new_page(viewport={"width": width, "height": 650}, device_scale_factor=1)
        page.set_content('<html><head><meta charset="utf-8"><style>' +
                         'svg {width:100%;height:auto;display:block}</style></head>' +
                         '<body style="margin:0"><div style="width:100%">' + text + '</div></body></html>')
        page.evaluate("document.fonts.ready")
        target = ART / "figures" / f"clue-position-{width}.png"
        page.locator("svg").screenshot(path=str(target))
        measurements.append({"render": target.name, "width": width,
                             "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                             "svg_box": page.locator("svg").bounding_box()})
        page.close()
    result = {"browser": browser.version, "python": sys.version, "platform": platform.platform(),
              "device": "cpu", "source_svg_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
              "rendered": measurements, "scope": "Standalone unchanged SVG; no course page layout test."}
    browser.close()
(ART / "figures/render-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False))
