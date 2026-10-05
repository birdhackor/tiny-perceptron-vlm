import asyncio,json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
from playwright.async_api import async_playwright
BASE=Path('/workspace/tiny-perceptron-vlm/docs/course-revision-20261005/continuity/artifacts/01_05/recheck-01')
async def run():
 p=sys.argv[1]
 async with async_playwright() as pw:
  b=await pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
  for label,w,h in [('desktop',1280,900),('mobile',390,844)]:
   page=await b.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
   url=f'http://127.0.0.1:8765/{p}.html'
   await page.goto(url,wait_until='domcontentloaded',timeout=15000)
   await page.evaluate('document.fonts.ready')
   imgs=page.locator('main img')
   count=await imgs.count()
   if not count:
    imgs=page.locator('article img'); count=await imgs.count()
   for i in range(count):
    im=imgs.nth(i); src=await im.get_attribute('src')
    await im.evaluate('(el)=>window.scrollTo(0,Math.max(0,el.getBoundingClientRect().top+window.scrollY-110))')
    await page.wait_for_timeout(150)
    out=BASE/f'{p}-{label}-{i+1}.png'
    await page.screenshot(path=str(out),full_page=False,timeout=15000)
    rec={'page_id':p,'url':url,'viewport':{'width':w,'height':h},'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'image_src':src,'image_box':await im.bounding_box(),'screenshot':str(out.relative_to(Path('/workspace/tiny-perceptron-vlm'))),'screenshot_absolute':str(out),'screenshot_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'viewed':False}
    with (BASE/'pageviews.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    print(json.dumps(rec,ensure_ascii=False))
    box=await im.bounding_box()
    if box['y']+box['height']>h-20:
     await page.evaluate('(amount)=>window.scrollBy(0,amount)',box['y']+box['height']-h+90)
     await page.wait_for_timeout(100)
     lower=BASE/f'{p}-{label}-{i+1}-lower.png'
     await page.screenshot(path=str(lower),full_page=False,timeout=15000)
     rec.update(recorded_at_utc=datetime.now(timezone.utc).isoformat(),image_box=await im.bounding_box(),screenshot=str(lower.relative_to(Path('/workspace/tiny-perceptron-vlm'))),screenshot_absolute=str(lower),screenshot_sha256=hashlib.sha256(lower.read_bytes()).hexdigest())
     with (BASE/'pageviews.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
     print(json.dumps(rec,ensure_ascii=False))
   await page.close()
  await b.close()
asyncio.run(run())
