"""Bounded page inspection; only the affected first Python code block is captured."""
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

A = Path(__file__).resolve().parent
async def main():
    result = {'url':'http://127.0.0.1:8765/13.14.html', 'browser':'/usr/bin/chromium'}
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(executable_path='/usr/bin/chromium', headless=True,
                args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage'],timeout=12000)
            page = await browser.new_page(viewport={'width':1280,'height':800})
            response = await page.goto(result['url'],wait_until='domcontentloaded',timeout=12000)
            result['http_status'] = response.status
            pres = page.locator('pre')
            result['pre_count'] = await pres.count()
            first = pres.first
            result['first_code_text'] = await first.inner_text(timeout=5000)
            await first.screenshot(path=str(A/'first-fence-desktop.png'),timeout=5000)
            await page.set_viewport_size({'width':390,'height':844})
            await first.screenshot(path=str(A/'first-fence-mobile.png'),timeout=5000)
            result['rendered'] = True
            await browser.close()
    except Exception as e:
        result.update(rendered=False,error=type(e).__name__+': '+str(e))
    (A/'page-render-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
asyncio.run(main())
