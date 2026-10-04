from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
ROOT=Path('/workspace/tiny-perceptron-vlm'); OUT=Path(__file__).resolve().parent
names=['natural-v4-family-crops.svg','natural_base_adapter.svg','natural-v4-answer-mask.svg','natural_photo_evidence.svg','natural_reading_order.svg','natural-v4-asr-two-routes.svg']
receipts=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium')
    page=browser.new_page(viewport={'width':1280,'height':1800},device_scale_factor=1)
    for name in names:
        source=ROOT/'course/figures'/name; raw=source.read_bytes(); snapshot=name+'.snapshot'; (OUT/snapshot).write_bytes(raw)
        url='http://127.0.0.1:8769/figures/'+name
        response=page.goto(url,wait_until='networkidle'); retrieved=response.body(); (OUT/(name+'.browser.bytes')).write_bytes(retrieved)
        measured=page.locator('svg').evaluate('''(svg)=>{const b=svg.viewBox.baseVal;const ratio=b.height/b.width;svg.setAttribute('width','1180');svg.setAttribute('height',String(Math.round(1180*ratio)));svg.style.width='1180px';svg.style.height=String(Math.round(1180*ratio))+'px';return {viewBox:{x:b.x,y:b.y,width:b.width,height:b.height},rendered_width:1180,rendered_height:Math.round(1180*ratio)};}''')
        fn=name+'.render.png'; page.locator('svg').screenshot(path=str(OUT/fn))
        receipts.append({'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(raw).hexdigest(),'source_snapshot':snapshot,'browser_url':url,'http_status':response.status,'retrieved_sha256':hashlib.sha256(retrieved).hexdigest(),'same_source_bytes':raw==retrieved,'render_png':fn,'render':measured,'renderer':'Chromium '+browser.version,'render_method':'Playwright top-level SVG browser load, SVG element displayed at width 1180 with proportional viewBox height, locator screenshot; original file unchanged'})
        print(json.dumps(receipts[-1],ensure_ascii=False))
    (OUT/'figure-render-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n'); browser.close()
