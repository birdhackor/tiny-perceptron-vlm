"""Actual current 9.2 course page inspection, without changing site code."""
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

B=Path(__file__).resolve().parent
receipt={'url':'http://127.0.0.1:8765/9.2.html','scope':'Current 9.2 page only; no figures referenced. Screenshot supports current material visibility, not a separate independent-reader verdict.','source_sha256':hashlib.sha256((B/'current-section.md').read_bytes()).hexdigest(),'screenshots':[]}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
    receipt['chromium_version']=browser.version
    for label,w,h in [('desktop',1280,800),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
        response=page.goto(receipt['url'],wait_until='networkidle',timeout=20000)
        assert response and response.status==200
        page.locator('details').evaluate_all('(els)=>els.forEach(e=>e.open=true)')
        visible=page.locator('body').inner_text()
        assert '欠什麼' in visible and '適當澄清' in visible
        assert '能確定球數嗎' in visible
        page.screenshot(path=str(B/f'current-{label}.png'),full_page=True)
        if label=='desktop':
            (B/'actual-page.html').write_text(page.content())
            (B/'actual-page-visible.txt').write_text(visible)
        receipt['screenshots'].append({'label':label,'viewport':[w,h],'status':response.status,'png_sha256':hashlib.sha256((B/f'current-{label}.png').read_bytes()).hexdigest()})
        page.close()
    browser.close()
(B/'current-render-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
