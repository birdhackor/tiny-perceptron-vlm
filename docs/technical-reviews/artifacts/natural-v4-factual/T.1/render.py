"""Render canonical local SVG bytes in Chromium; no website/preview claim."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[5]
DEST=Path(__file__).resolve().parent
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    renders=[]
    for name,width,height in [('multimodal_wave_cycle',600,460),('fsdd_speaker_holdout',1120,720)]:
        path=ROOT/'course/figures'/f'{name}.svg'
        data=path.read_bytes()
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        page.set_content(data.decode())
        page.screenshot(path=str(DEST/f'{name}.png'))
        renders.append({'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(data).hexdigest(),
                        'render':str((DEST/f'{name}.png').relative_to(ROOT)),'viewport':[width,height]})
        page.close()
    print(json.dumps({'browser':browser.version,'renders':renders},indent=2))
    browser.close()
