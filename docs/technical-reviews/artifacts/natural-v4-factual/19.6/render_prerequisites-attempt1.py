"""Render the three necessary prerequisite SVGs unchanged in installed Chromium."""
from pathlib import Path
import hashlib, json
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[5]
ART=ROOT/'docs/technical-reviews/artifacts/natural-v4-factual/19.6'
figures=[('capstone_pipeline.svg',920,610),('multimodal_expand_image.svg',900,400),('multimodal_audio_axes.svg',600,700)]
receipts=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    for name,width,height in figures:
        path=ROOT/'course/figures'/name
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        page.goto(path.as_uri(),wait_until='load');page.locator('svg').screenshot(path=str(ART/(name+'.png')))
        receipts.append({'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'render':str((ART/(name+'.png')).relative_to(ROOT)),'viewport':[width,height]})
        page.close()
    print('CHROMIUM',browser.version)
    browser.close()
(ART/'render-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps(receipts,indent=2))
