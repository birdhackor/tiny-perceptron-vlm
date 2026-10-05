"""Render the necessary B.3 predecessor's unmodified SVG at 640 and 390 px."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parents[1]
FIGURE = BASE / "inputs/course/figures/rewrite-B-tool-flow.svg"
raw = FIGURE.read_bytes()
svg = raw.decode("utf-8")
renderings = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox", "--disable-gpu"])
    for width in [640, 390]:
        page = browser.new_page(viewport={"width": width, "height": 900}, device_scale_factor=1)
        page.set_content('<html lang="zh-Hant"><head><meta charset="utf-8"><style>body{margin:0;background:white}main{width:100%;max-width:640px}svg{display:block;width:100%;height:auto}</style></head><body><main>' + svg + '</main></body></html>')
        page.evaluate("document.fonts.ready")
        output = BASE / "checks" / f"predecessor-figure-{width}.png"
        page.locator("main").screenshot(path=str(output))
        renderings.append({"path": output.as_posix(), "width": width, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        page.close()
    browser.close()
receipt = {"source_path": "course/figures/rewrite-B-tool-flow.svg", "source_sha256": hashlib.sha256(raw).hexdigest(),
           "scope": "Necessary B.3 predecessor only; B.4 has no figure references. Actual Chromium rendering of original SVG bytes embedded with responsive width CSS.",
           "environment": {"python": platform.python_version(), "renderer": subprocess.check_output(["/usr/bin/chromium", "--version"], text=True).strip()},
           "renderings": renderings}
(BASE / "checks/figure-render-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
