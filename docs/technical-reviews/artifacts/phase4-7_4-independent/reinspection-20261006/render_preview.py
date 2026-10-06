"""Render the provided current local preview, preserving actual screenshots and DOM evidence."""
from pathlib import Path
import json
import hashlib
import subprocess
from datetime import UTC, datetime
from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parent
URL='http://127.0.0.1:8765/7.4.html'
results=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'],timeout=20000)
    for name,width,height in [('desktop',1280,800),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        response=page.goto(URL,wait_until='networkidle',timeout=20000)
        assert response.status==200
        page.locator('img').first.wait_for(state='visible',timeout=10000)
        raw=page.content().encode()
        text=page.locator('body').inner_text()
        assert '接著只把答案A改成B' in text
        assert '首答案預測位置' in text and '有效目標數對，不保證位置對' in text
        (OUT/(name+'.html')).write_bytes(raw)
        (OUT/(name+'.body.txt')).write_text(text)
        page.screenshot(path=str(OUT/(name+'.png')),full_page=True,timeout=10000)
        results.append({'name':name,'viewport':{'width':width,'height':height},'url':URL,'http_status':response.status,
          'screenshot':name+'.png','html':name+'.html','body_text':name+'.body.txt',
          'images':page.locator('img').evaluate_all('(imgs)=>imgs.map(x=>({src:x.getAttribute("src"),complete:x.complete,naturalWidth:x.naturalWidth,naturalHeight:x.naturalHeight,rect:{x:x.getBoundingClientRect().x,y:x.getBoundingClientRect().y,width:x.getBoundingClientRect().width,height:x.getBoundingClientRect().height}}))'),
          'body_scroll_width':page.locator('body').evaluate('(x)=>x.scrollWidth'),'viewport_width':width,
          'html_sha256':hashlib.sha256(raw).hexdigest(),'screenshot_sha256':hashlib.sha256((OUT/(name+'.png')).read_bytes()).hexdigest()})
        page.close()
    browser.close()
(OUT/'preview-render-receipt.json').write_text(json.dumps({'rendered_at_utc':datetime.now(UTC).isoformat(),'results':results},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
