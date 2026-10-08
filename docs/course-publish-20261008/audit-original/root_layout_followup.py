from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from playwright.async_api import async_playwright
import asyncio, threading, json, hashlib
from datetime import datetime, timezone

B = Path(__file__).resolve().parent
SITE = Path('/workspace/tiny-perceptron-vlm/outputs/tutorial-audit-20261008/site')
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args): pass
server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(SITE)))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = 'http://127.0.0.1:' + str(server.server_port) + '/'

async def main():
    records = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path='/usr/bin/chromium', args=['--no-sandbox'])
        page = await browser.new_page(viewport={'width':390,'height':844}, reduced_motion='reduce')
        await page.goto(base+'first-steps.html#W.1',wait_until='domcontentloaded')
        await page.evaluate('async()=>{await document.fonts.ready}')
        await page.locator('[id="W.1"]').evaluate("e=>e.scrollIntoView({block:'start'})")
        await page.evaluate('window.scrollBy(0,180)')
        out = B/'page-captures/first-steps-390-W1-scroll-followup.png'
        await page.screenshot(path=str(out),animations='disabled')
        records.append({'page_id':'first-steps','operation':'ordinary vertical scroll 180px after W.1 anchor','path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
        await page.goto(base+'16.7.html',wait_until='domcontentloaded')
        await page.evaluate('async()=>{await document.fonts.ready}')
        details=page.locator('article details').first
        await details.locator('summary').click()
        table=details.locator('table').first
        await table.evaluate("e=>e.scrollIntoView({block:'start'})")
        state=await table.evaluate("""e=>{let p=e;while(p && !(p.scrollWidth>p.clientWidth+2 && ['auto','scroll'].includes(getComputedStyle(p).overflowX)))p=p.parentElement;if(!p)return {found:false};let before=p.scrollLeft;p.scrollLeft=p.scrollWidth-p.clientWidth;return {found:true,tag:p.tagName,class:p.className,before,after:p.scrollLeft,scrollWidth:p.scrollWidth,clientWidth:p.clientWidth}}""")
        out=B/'page-captures/16.7-390-details-table-right-followup.png'
        await page.screenshot(path=str(out),animations='disabled')
        records.append({'page_id':'16.7','operation':'open details then horizontal-scroll comparison table to right edge','state':state,'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
        await browser.close()
    (B/'checks/root-layout-followup-captures.json').write_text(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'capture_only_not_visual_verdict':True,'records':records},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(records,ensure_ascii=False,indent=2))
try: asyncio.run(main())
finally: server.shutdown();server.server_close()
