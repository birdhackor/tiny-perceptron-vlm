"""Actually render the fixed current course preview for owner reinspection."""
from pathlib import Path
import hashlib
import json
from playwright.sync_api import sync_playwright

DEST=Path(__file__).resolve().parent
URL='http://127.0.0.1:8765/13.12.html'
records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
                             args=['--no-sandbox','--disable-dev-shm-usage'],timeout=25000)
    for width,height in [(1280,800),(390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        response=page.goto(URL,wait_until='load',timeout=25000)
        assert response and response.status==200
        page.locator('h1,h2').filter(has_text='13.12').first.wait_for(timeout=10000)
        page.screenshot(path=str(DEST/f'preview-{width}x{height}.png'),full_page=True,timeout=25000)
        page.locator('details').evaluate_all('(items)=>items.forEach(x=>x.open=true)')
        page.screenshot(path=str(DEST/f'preview-expanded-{width}x{height}.png'),full_page=True,timeout=25000)
        text=page.locator('body').inner_text()
        assert '收集這批回答時' in text and '更新前後的機率比' in text
        html=page.content().encode()
        (DEST/f'preview-{width}x{height}.html').write_bytes(html)
        records.append({'url':URL,'status':response.status,'viewport':[width,height],
                        'html_sha256':hashlib.sha256(html).hexdigest(),
                        'scroll_width':page.evaluate('document.documentElement.scrollWidth'),
                        'client_width':page.evaluate('document.documentElement.clientWidth'),
                        'details_count':page.locator('details').count(),
                        'image_count':page.locator('img').count(),
                        'expanded_text_sha256':hashlib.sha256(text.encode()).hexdigest()})
        page.close()
    browser.close()
(DEST/'preview-render.result.json').write_text(json.dumps({'browser':'Chromium 151.0.7922.173','playwright':'1.63.0','renders':records},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False,indent=2))
