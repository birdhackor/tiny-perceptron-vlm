from pathlib import Path
import json
from playwright.sync_api import sync_playwright
OUT=Path('/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-supplemental/student')
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    page.goto('http://127.0.0.1:8769/natural-v4-student.html',wait_until='networkidle')
    receipts=[]
    for i,pre in enumerate(page.locator('article pre').all()):
        pre.scroll_into_view_if_needed()
        result=pre.evaluate('''(p)=>[p,...p.querySelectorAll('*')].map(e=>{
            const before=e.scrollLeft; e.scrollLeft=e.scrollWidth;
            return {tag:e.tagName,class:e.className,clientWidth:e.clientWidth,scrollWidth:e.scrollWidth,before,after:e.scrollLeft,overflowX:getComputedStyle(e).overflowX};
        }).filter(e=>e.after>0)''')
        pre.screenshot(path=str(OUT/'browser'/f'code-block-{i:02d}-actually-scrolled.png'))
        receipts.append({'index':i,'actual_scrolled_elements':result,'command':pre.inner_text()})
    (OUT/'browser/code-horizontal-scroll-receipt.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipts,ensure_ascii=False,indent=2))
    browser.close()
