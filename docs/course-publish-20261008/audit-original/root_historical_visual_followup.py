from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from playwright.async_api import async_playwright
import asyncio, threading, json, hashlib

B=Path(__file__).resolve().parent
SITE=Path('/workspace/tiny-perceptron-vlm/outputs/tutorial-audit-20261008/site')
class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args): pass
server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(SITE)))
threading.Thread(target=server.serve_forever,daemon=True).start()
async def main():
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
        page=await browser.new_page(viewport={'width':390,'height':844},reduced_motion='reduce')
        await page.goto(f'http://127.0.0.1:{server.server_port}/9.8.html',wait_until='domcontentloaded')
        await page.evaluate('async()=>{await document.fonts.ready}')
        await page.locator('article details').evaluate_all('es=>es.forEach(e=>e.open=true)')
        table=page.locator('article table').nth(1)
        await table.evaluate("e=>e.scrollIntoView({block:'center'})")
        await page.wait_for_timeout(200)
        out=B/'page-captures/9.8-390-historical-optional-table-open-followup.png'
        await page.screenshot(path=str(out),animations='disabled')
        print(json.dumps({'capture_only':True,'old_prompt':'B-9.8-mobile-prompt-wrap after current cross reports sealed','path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'viewport':[390,844]},ensure_ascii=False))
        await browser.close()
try:asyncio.run(main())
finally:server.shutdown();server.server_close()
