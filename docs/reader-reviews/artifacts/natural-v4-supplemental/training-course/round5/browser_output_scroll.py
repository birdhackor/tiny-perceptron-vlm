import json,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
q=Path('docs/reader-reviews/artifacts/natural-v4-supplemental/training-course/round5')
with sync_playwright() as w:
 b=w.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
 response=page.goto('http://127.0.0.1:8789/16.1.html',wait_until='networkidle')
 assert response.body()==(q/'16.1.http-response.html').read_bytes()
 pre=page.locator('article .output pre').first;pre.scroll_into_view_if_needed()
 states=pre.evaluate('(e)=>{const before={scrollLeft:e.scrollLeft,clientWidth:e.clientWidth,scrollWidth:e.scrollWidth};e.scrollLeft=e.scrollWidth;return{before,after:{scrollLeft:e.scrollLeft,clientWidth:e.clientWidth,scrollWidth:e.scrollWidth}}}')
 page.wait_for_timeout(80);fn=q/'16.1-CPU-output-right-columns.png';page.screenshot(path=str(fn))
 (q/'CPU-output-horizontal-scroll-receipt.json').write_text(json.dumps({'actual_browser':'Chromium /usr/bin/chromium','URL':page.url,'exact_previous_HTTP_bytes':True,'actual_scroll':states,'screenshot':str(fn),'purpose':'personally expose CPU total/avg and call count columns of the real already-executed root profiler widget; no numerical execution'},ensure_ascii=False,indent=2)+'\n')
 b.close()
print(json.dumps(states))
