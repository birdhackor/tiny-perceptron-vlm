from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from playwright.async_api import async_playwright
from datetime import datetime, timezone
import asyncio, json, threading, hashlib

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
        page = await browser.new_page(reduced_motion='reduce')
        for width, height in [(1280, 800), (390, 844)]:
            await page.set_viewport_size({'width': width, 'height': height})
            for pid in ['7.4', '16.7']:
                await page.goto(base + pid + '.html', wait_until='domcontentloaded')
                await page.evaluate('async()=>{await document.fonts.ready}')
                details = page.locator('article details').first
                summary = details.locator('summary').first
                before = await details.evaluate('e=>e.open')
                await summary.click()
                after = await details.evaluate('e=>({open:e.open,text:e.innerText,overflow:document.documentElement.scrollWidth>innerWidth+1})')
                await summary.evaluate("e=>e.scrollIntoView({block:'start'})")
                out = B / 'page-captures' / f'{pid}-{width}-details-open.png'
                await page.screenshot(path=str(out), animations='disabled')
                await summary.click()
                closed = await details.evaluate('e=>e.open')
                records.append({'page_id':pid, 'viewport':[width,height], 'before_open':before, 'after_open':after, 'after_close':closed, 'path':str(out), 'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
            await page.goto(base + 'first-steps.html#W.1', wait_until='domcontentloaded')
            await page.evaluate('async()=>{await document.fonts.ready}')
            await page.locator('[id="W.1"]').evaluate("e=>e.scrollIntoView({block:'start'})")
            out = B / 'page-captures' / f'first-steps-{width}-W1-viewport.png'
            await page.screenshot(path=str(out), animations='disabled')
            records.append({'page_id':'first-steps','viewport':[width,height],'operation':'navigate to actual W.1 anchor and scroll heading into viewport','path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
        await browser.close()
    (B/'checks/interaction-captures.json').write_text(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'scope':'actual local source-preview interactions; screenshots and DOM are not themselves a human visual verdict','records':records},ensure_ascii=False,indent=2)+'\n')
    print('Captured six interaction/anchor viewport records.')

try:
    asyncio.run(main())
finally:
    server.shutdown(); server.server_close()
