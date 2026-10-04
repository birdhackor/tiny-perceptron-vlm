import hashlib, json, platform
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
svg=ROOT/'course/figures/natural_reading_order.svg'
results=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    for width in [720,358]:
        height=round(width*1160/720)
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        page.set_content('<html><body style="margin:0">'+svg.read_text().replace('viewBox="0 0 720 1160"',f'width="{width}" height="{height}" viewBox="0 0 720 1160"')+'</body></html>')
        page.evaluate('document.fonts.ready')
        path=OUT/f'natural-final-fact-20.6-render-{width}.png'
        page.screenshot(path=str(path))
        results.append({'width':width,'height':height,'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'browser_version':browser.version,'device_scale_factor':1,'source_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'font':page.evaluate('getComputedStyle(document.querySelector("text")).fontFamily')})
        page.close()
    browser.close()
print(json.dumps({'platform':platform.platform(),'renders':results},indent=2))
