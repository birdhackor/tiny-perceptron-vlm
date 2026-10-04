from pathlib import Path
import hashlib
import json
from playwright.sync_api import sync_playwright
ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/reader-reviews/artifacts/natural-v4-supplemental/student'
BASE='http://127.0.0.1:8769/'
names=['natural-v4-student.html','index.html','course.html','chapter-20.html','20.1.html','20.2.html','20.3.html','20.9.html','20.10.html','20.11.html','20.12.html','20.13.html','natural-v4-data.html','natural-v4-training.html']
receipts=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    for name in names:
        response=page.goto(BASE+name,wait_until='networkidle')
        raw=response.body()
        path=OUT/'browser'/f'{name}.http-raw.html'
        path.write_bytes(raw)
        receipts.append({'url':page.url,'status':response.status,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'original_response_body':str(path.relative_to(ROOT))})
        if name=='course.html':
            for i,heading in enumerate(page.locator('article h2').all()):
                heading.scroll_into_view_if_needed()
                page.screenshot(path=str(OUT/'browser'/f'course-section-{i}.png'))
        if name=='20.1.html':
            page.locator('article pre').first.scroll_into_view_if_needed()
            page.screenshot(path=str(OUT/'browser/20.1-code-with-prose.png'))
        if name=='natural-v4-student.html':
            page.locator('article table').first.scroll_into_view_if_needed()
            page.screenshot(path=str(OUT/'browser/student-measurement-table.png'))
    (OUT/'browser/http-raw-receipt.json').write_text(json.dumps({'engine':'Chromium '+browser.version,'receipts':receipts},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pages':len(receipts),'all_statuses':[r['status'] for r in receipts],'receipt':str((OUT/'browser/http-raw-receipt.json').relative_to(ROOT))},ensure_ascii=False))
    browser.close()
