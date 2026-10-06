from pathlib import Path
import json
from playwright.sync_api import sync_playwright
out=Path('/workspace/selftrained-v2/docs/technical-reviews/artifacts/p6-19.1/evidence')
svg=Path('/workspace/selftrained-v2/course/figures/p6-19-start-core.svg').read_text()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
    rows=[]
    for width,height in [(640,770),(360,434)]:
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        page.set_content('<html><head><style>html,body{margin:0}svg{display:block;width:100%;height:auto}</style></head><body>'+svg+'</body></html>',wait_until='load',timeout=20000)
        page.screenshot(path=str(out/f'figure-{width}.png'))
        rows.append({'width':width,'height':height,'browser':browser.version,'method':'inline original SVG; responsive width 100%; device_scale_factor=1'})
        page.close()
    browser.close()
print(json.dumps(rows,indent=2))
