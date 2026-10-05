from pathlib import Path
import hashlib,json
from playwright.sync_api import sync_playwright
out=Path(__file__).resolve().parent
svg=out/"frozen-input/course/figures/rewrite-B-tool-flow.svg"
raw=svg.read_text()
with sync_playwright() as playwright:
    browser=playwright.chromium.launch(headless=True,executable_path="/usr/bin/chromium")
    page=browser.new_page(viewport={"width":1280,"height":1000},device_scale_factor=1)
    page.set_content('<html><body style="margin:0">'+raw+'</body></html>')
    page.locator("svg").screenshot(path=str(out/"figure-intrinsic.png"))
    intrinsic=page.locator("svg").bounding_box()
    page.set_viewport_size({"width":390,"height":844})
    page.set_content('<html><head><style>body {margin:0} svg {display:block;width:100%;height:auto}</style></head><body>'+raw+'</body></html>')
    page.screenshot(path=str(out/"figure-mobile.png"),full_page=True)
    print(json.dumps({"browser_version":browser.version,"svg_sha256":hashlib.sha256(svg.read_bytes()).hexdigest(),"intrinsic":intrinsic,"mobile_figure_box":page.locator("svg").bounding_box(),"screenshots":["figure-intrinsic.png","figure-mobile.png"]}))
    browser.close()
