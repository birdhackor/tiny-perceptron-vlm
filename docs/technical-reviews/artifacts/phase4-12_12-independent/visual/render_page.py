"""Render the current section with bounded browser waits, and check served bytes."""
import asyncio
import hashlib
import json
import pathlib
import urllib.request
from playwright.async_api import async_playwright

BASE=pathlib.Path(__file__).parent
ROOT=BASE.resolve().parents[4]
URL='http://127.0.0.1:8765/12.12.html'

async def main():
    results=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
                args=['--no-sandbox','--disable-dev-shm-usage'],timeout=15000)
        for name,width,height in [('desktop',1280,800),('mobile',390,844)]:
            context=await browser.new_context(viewport={'width':width,'height':height},device_scale_factor=1)
            await context.route('https://**/*',lambda route:route.abort())
            page=await context.new_page()
            await page.goto(URL,wait_until='domcontentloaded',timeout=15000)
            await page.locator('main article h1').wait_for(timeout=5000)
            await page.screenshot(path=str(BASE/f'page-{name}-playwright.png'),full_page=True,timeout=15000)
            code=await page.locator('main article pre code').inner_text()
            original=(ROOT/'docs/technical-reviews/artifacts/phase4-12_12-independent/execution/fence-1.py').read_text()
            assert code.rstrip('\n')==original.rstrip('\n')
            texts=await page.locator('main article p').all_text_contents()
            expected=['圖片是紅圓，聲音是 440 Hz 單音。','驗證集只有 8/12']
            assert all(any(x in t for t in texts) for x in expected)
            img=page.locator('main article img.diagram')
            assert await img.evaluate('(x)=>x.complete && x.naturalWidth>0')
            (BASE/f'page-{name}-article.txt').write_text(await page.locator('main article').inner_text())
            results.append({'viewport':[width,height],'browser':browser.version,'source_fence_equal':True,
                            'image_loaded':True,'diagram_rectangle':await img.bounding_box()})
            await context.close()
        await browser.close()
    served=urllib.request.urlopen('http://127.0.0.1:8765/figures/rewrite-12-joint-clues.svg',timeout=10).read()
    local=(ROOT/'course/figures/rewrite-12-joint-clues.svg').read_bytes()
    assert served==local
    result={'url':URL,'screens':results,'served_figure_sha256':hashlib.sha256(served).hexdigest(),
            'external_resource_policy':'External HTTPS requests aborted to bound optional font/analytics waits; local content unchanged.'}
    (BASE/'render-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

asyncio.run(main())
