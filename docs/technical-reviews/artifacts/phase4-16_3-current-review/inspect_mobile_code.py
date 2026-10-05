import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
P=Path(__file__).resolve().parent
async def main():
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,timeout=10000,args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking','--no-first-run'])
  page=await browser.new_page(viewport={'width':390,'height':844},device_scale_factor=1)
  await page.goto('http://127.0.0.1:8765/16.3.html',wait_until='load',timeout=10000)
  await page.evaluate('document.fonts.ready')
  pres=page.locator('pre')
  print(json.dumps({'pre_blocks':await pres.count(),'browser':browser.version}),flush=True)
  for i in range(await pres.count()):
   pre=pres.nth(i)
   metrics=await pre.evaluate('(e)=>({clientWidth:e.clientWidth,scrollWidth:e.scrollWidth,text:e.innerText})')
   print(json.dumps({'pre_index':i,'metrics':metrics},ensure_ascii=False),flush=True)
   if i==0:
    await pre.evaluate('(e)=>{e.scrollLeft=e.scrollWidth;e.scrollIntoView({block:"center"});}')
    await page.screenshot(path=str(P/'execution/mobile-code-right.png'),timeout=10000)
   else:
    await pre.scroll_into_view_if_needed()
    await page.screenshot(path=str(P/'execution/mobile-output.png'),timeout=10000)
  await browser.close()
asyncio.run(main())
