import json
import platform
from pathlib import Path
from playwright.sync_api import sync_playwright

base=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':640,'height':482},device_scale_factor=1)
    page.set_content('<html><body style="margin:0">'+(base/'inputs/policy-update.svg').read_text()+'</body></html>')
    page.evaluate('document.fonts.ready')
    page.screenshot(path=str(base/'checks/policy-update.render.png'))
    result={'browser':browser.version,'python':platform.python_version(),'svg':str(base/'inputs/policy-update.svg'),
            'viewport':[640,482],'result':'Native-size SVG rendered in Chromium; screenshot requires reviewer visual inspection.'}
    browser.close()
(base/'checks/render.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
