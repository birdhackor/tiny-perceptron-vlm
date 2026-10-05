import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
P=Path(__file__).resolve().parent
async def main():
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,timeout=10000,args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking','--no-first-run'])
  page=await browser.new_page(viewport={'width':390,'height':844},device_scale_factor=1)
  await page.goto('http://127.0.0.1:8765/16.3.html',wait_until='load',timeout=10000)
  pre=page.locator('pre').first
  rows=await pre.evaluate('''e=>{let a=[e,...e.querySelectorAll('*')];let p=e.parentElement;for(let i=0;p&&i<4;i++,p=p.parentElement)a.push(p);return a.filter((x,i)=>i===0||x.scrollWidth>x.clientWidth+2||i>=a.length-4).map(x=>({tag:x.tagName,class:x.className,client:x.clientWidth,scroll:x.scrollWidth,overflow:getComputedStyle(x).overflowX}));}''')
  print(json.dumps({'layout':rows}),flush=True)
  result=await pre.evaluate('''e=>{let a=[e,...e.querySelectorAll('*')];let p=e.parentElement;for(let i=0;p&&i<4;i++,p=p.parentElement)a.push(p);let x=a.find(x=>x.clientWidth>0&&x.scrollWidth>x.clientWidth+2&&['auto','scroll'].includes(getComputedStyle(x).overflowX));if(!x)return {found:false};x.scrollLeft=x.scrollWidth;x.scrollIntoView({block:'center'});return {found:true,tag:x.tagName,class:x.className,client:x.clientWidth,scroll:x.scrollWidth,left:x.scrollLeft};}''')
  print(json.dumps({'local_code_scroll':result}),flush=True)
  await page.screenshot(path=str(P/'execution/mobile-code-actual-right.png'),timeout=10000)
  await browser.close()
asyncio.run(main())
