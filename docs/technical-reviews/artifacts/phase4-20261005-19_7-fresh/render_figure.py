import importlib.metadata
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ART = Path(__file__).resolve().parent
svg = (ART / "inputs/rewrite-19-tool-roundtrip.svg").read_text()
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    version = browser.version
    for width, name in [(640, "figure-intrinsic.png"), (390, "figure-mobile.png")]:
        page = browser.new_page(viewport={"width": width, "height": 820}, device_scale_factor=1)
        page.set_content('<html><head><meta charset="utf-8"><style>body{margin:0}svg{display:block;width:100%;height:auto}</style></head><body>'+svg+'</body></html>')
        page.locator("svg").screenshot(path=str(ART / name))
        page.close()
    browser.close()
(ART / "figure-render-environment.json").write_text(json.dumps({"browser": version, "playwright": importlib.metadata.version("playwright"), "viewports": [640,390], "source": "inputs/rewrite-19-tool-roundtrip.svg"}, indent=2)+'\n')
print("Rendered actual frozen SVG at 640 px and 390 px with Chromium", version)
