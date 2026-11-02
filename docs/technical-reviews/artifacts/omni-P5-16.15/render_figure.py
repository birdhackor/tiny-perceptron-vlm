"""Render the actual SVG; screenshots stay outside Git."""

import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[4]
OUTSIDE = Path("/workspace/work/reference/omni-P5-16.15")
FIGURE = ROOT / "course/figures/omni-16.15-readonly-memory.svg"
BROWSER = "/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell"

screenshots = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=BROWSER, headless=True, args=["--no-sandbox"])
    for width, height in [(1280, 800), (390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        page.set_content(
            '<html lang="zh-Hant"><meta charset="utf-8"><style>body{margin:16px;background:white}svg{display:block;width:100%;max-width:560px;height:auto}</style>'
            + FIGURE.read_text()
            + "</html>"
        )
        page.evaluate("document.fonts.ready")
        target = OUTSIDE / f"figure-{width}x{height}.png"
        page.screenshot(path=str(target), full_page=True)
        screenshots.append({"viewport": f"{width}x{height}", "path_outside_git": str(target), "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        page.close()
    browser.close()
receipt = {"figure": str(FIGURE.relative_to(ROOT)), "source_sha256": hashlib.sha256(FIGURE.read_bytes()).hexdigest(), "method": "Chromium standalone responsive SVG at native maximum 560px and width 358px. This checks diagram contents, not course page layout.", "screenshots": screenshots}
target = Path(__file__).resolve().parent / "figure-render-receipt-v1.json"
with target.open("x") as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(receipt, ensure_ascii=False))
