import hashlib,json,sys
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent
FIG=OUT/'repository-snapshots/course/figures/rewrite-19-shared-core.svg'
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':640,'height':669},device_scale_factor=1)
 page.set_content('<html><head><meta charset="utf-8"></head><body style="margin:0">'+FIG.read_text()+'</body></html>'); page.screenshot(path=str(OUT/'shared-core-render.png'))
 print(json.dumps({'source_sha256':hashlib.sha256(FIG.read_bytes()).hexdigest(),'renderer':browser.version,'python':sys.version.split()[0],'viewport':[640,669],'output':'shared-core-render.png','output_sha256':hashlib.sha256((OUT/'shared-core-render.png').read_bytes()).hexdigest()}))
 browser.close()
