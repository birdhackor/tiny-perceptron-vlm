from pathlib import Path
from playwright.sync_api import sync_playwright
import json
OUT=Path('/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-supplemental/training-course')
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox']);page=b.new_page(viewport={'width':1440,'height':1100})
 page.goto('http://127.0.0.1:8769/training.html',wait_until='networkidle')
 pre=page.locator('article pre').filter(has_text='scripts/train_simple.py --model bigram --seed 42 --device cpu --train').first
 pre.scroll_into_view_if_needed()
 candidates=pre.evaluate('(el)=>[el,el.firstElementChild,el.parentElement,el.parentElement.parentElement].map(x=>({tag:x.tagName,classes:x.className,scrollWidth:x.scrollWidth,clientWidth:x.clientWidth,overflowX:getComputedStyle(x).overflowX}))')
 scroll=pre.evaluate('(el)=>{ const nodes=[el,el.firstElementChild,el.parentElement,el.parentElement.parentElement]; const target=nodes.find(x=>x.scrollWidth>x.clientWidth); if(!target) return {found:false}; target.scrollLeft=target.scrollWidth; return {found:true,tag:target.tagName,classes:target.className,scrollLeft:target.scrollLeft,scrollWidth:target.scrollWidth,clientWidth:target.clientWidth}; }')
 page.screenshot(path=str(OUT/'long-command-actual-right.png'))
 r={'url':page.url,'rendered_command_text':pre.inner_text(),'candidates':candidates,'actual_scroll':scroll,'screenshot':'long-command-actual-right.png'};(OUT/'browser-code-scroll-receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False,indent=2));b.close()
