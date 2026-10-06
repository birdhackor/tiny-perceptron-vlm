from pathlib import Path
import json,hashlib,sys,importlib.metadata as md
from playwright.sync_api import sync_playwright
out=Path(__file__).resolve().parent
root=out.parents[3]
result={'environment':{'python':sys.version,'playwright':md.version('playwright'),'renderer':'local Chromium'},'renders':[]}
v=json.loads((out/'verification.json').read_text())
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu','--disable-background-networking'])
    result['environment']['chromium']=browser.version
    for name in ['fashion','ocr','voice']:
        for item in v['figures'][name]['renders']:
            width,height=item['viewport']
            page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
            svg=(out/'inputs'/f'p6-19-modal-{name}.svg').read_text()
            html=f'<html><head><meta charset="utf-8"><style>body{{margin:16px;background:white}}svg{{display:block;width:100%;max-width:560px;height:auto}}</style></head><body>{svg}</body></html>'
            (root/item['html']).write_text(html)
            page.set_content(html,wait_until='domcontentloaded',timeout=20000)
            page.locator('svg').wait_for(state='visible',timeout=20000)
            page.screenshot(path=str(root/item['png']),full_page=True,timeout=20000)
            raw=(root/item['png']).read_bytes()
            result['renders'].append({**item,'sha256':hashlib.sha256(raw).hexdigest(),'image_box':page.locator('svg').bounding_box()})
            page.close()
    browser.close()
(out/'render-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
