import asyncio
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from playwright.async_api import async_playwright

P=Path(__file__).resolve().parent
URL='http://127.0.0.1:8765/16.3.html'
with urlopen(URL,timeout=10) as response:
    raw=response.read()
    (P/'input/current-page.html').write_bytes(raw)
    print(json.dumps({'url':URL,'status':response.status,'page_sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}),flush=True)
async def main():
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,timeout=10000,args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking','--disable-component-update','--no-first-run'])
        print(json.dumps({'browser_version':browser.version}),flush=True)
        for label,width,height in [('desktop',1280,800),('mobile',390,844)]:
            page=await browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
            await page.goto(URL,wait_until='load',timeout=10000)
            await page.evaluate('Promise.race([document.fonts.ready, new Promise(r => setTimeout(r, 1500))])')
            images=await page.locator('img').evaluate_all('(els)=>els.map(e=>({src:e.getAttribute("src"),complete:e.complete,naturalWidth:e.naturalWidth,naturalHeight:e.naturalHeight,rect:{x:e.getBoundingClientRect().x,y:e.getBoundingClientRect().y,width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height}}))')
            dimensions=await page.evaluate('({client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight})')
            await page.screenshot(path=str(P/f'execution/{label}-full-page.png'),full_page=True,timeout=10000)
            await page.screenshot(path=str(P/f'execution/{label}-top.png'),timeout=10000)
            figure=page.locator('img').first
            await figure.scroll_into_view_if_needed()
            await page.screenshot(path=str(P/f'execution/{label}-figure.png'),timeout=10000)
            code=page.locator('pre').first
            await code.scroll_into_view_if_needed()
            await page.screenshot(path=str(P/f'execution/{label}-code.png'),timeout=10000)
            details=page.locator('details').first
            if await details.count():
                await details.evaluate('(e)=>e.open=true')
                await details.scroll_into_view_if_needed()
                await page.screenshot(path=str(P/f'execution/{label}-measurement.png'),timeout=10000)
            print(json.dumps({'viewport':label,'width':width,'height':height,'dimensions':dimensions,'images':images,'screenshots':['full-page','top','figure','code','measurement']},ensure_ascii=False),flush=True)
            await page.close()
        await browser.close()
asyncio.run(main())
