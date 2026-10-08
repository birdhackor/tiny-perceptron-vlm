from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.async_api import async_playwright
import asyncio,json,threading,hashlib
B=Path(__file__).resolve().parent
site=Path('/workspace/tiny-perceptron-vlm/outputs/tutorial-audit-20261008/site')
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(site)))
threading.Thread(target=server.serve_forever,daemon=True).start()
base='http://127.0.0.1:'+str(server.server_port)+'/'
ids=['1.3','3.4','7.4','10.5','11.17','12.5','13.14','14.1','15.5','15.6','16.3','16.4','16.14','17.8','18.6','19.6','20.7','B.3','C.7']
async def main():
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
  q=asyncio.Queue()
  for pid in ids:q.put_nowait(pid)
  async def worker():
   page=await browser.new_page(reduced_motion='reduce')
   while not q.empty():
    pid=await q.get()
    for width,height in [(1280,800),(390,844)]:
     await page.set_viewport_size({'width':width,'height':height})
     await page.goto(base+pid+'.html',wait_until='domcontentloaded')
     await page.evaluate('async()=>{await document.fonts.ready;await Promise.all([...document.images].map(i=>i.decode().catch(()=>{})))}')
     for i,img in enumerate(await page.locator('article img').all()):
      if not await img.is_visible():continue
      await img.evaluate("e=>e.scrollIntoView({block:'center'})")
      out=B/'page-captures'/f'{pid}-{width}-context-{i}.png'
      await page.screenshot(path=str(out),animations='disabled')
      record={'page_id':pid,'viewport':[width,height],'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'scope':'actual viewport at source figure, with naturally visible neighboring text; screenshot alone is not manual verification'}
      with (B/'checks/context-captures.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    q.task_done()
   await page.close()
  await asyncio.gather(*(worker() for _ in range(3)))
  await browser.close()
asyncio.run(main());server.shutdown();server.server_close()
