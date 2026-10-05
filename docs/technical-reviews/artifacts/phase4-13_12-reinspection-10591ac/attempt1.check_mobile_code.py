from pathlib import Path
from playwright.sync_api import sync_playwright
import json

D=Path(__file__).resolve().parent
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'],timeout=25000)
    page=b.new_page(viewport={'width':390,'height':844})
    page.goto('http://127.0.0.1:8765/13.12.html',wait_until='load',timeout=25000)
    pre=page.locator('article pre').first
    records=pre.evaluate('''(e)=>{const r=[];for(let i=0;i<4&&e;i++,e=e.parentElement){const s=getComputedStyle(e);r.push({tag:e.tagName,class:e.className,clientWidth:e.clientWidth,scrollWidth:e.scrollWidth,overflowX:s.overflowX});}return r}''')
    pre.evaluate('''(e)=>{while(e){if(e.scrollWidth>e.clientWidth && ['auto','scroll'].includes(getComputedStyle(e).overflowX)){e.scrollLeft=e.scrollWidth;break}e=e.parentElement}}''')
    pre.screenshot(path=str(D/'mobile-code-scrolled-right.png'),timeout=25000)
    b.close()
(D/'mobile-code-scroll.result.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
