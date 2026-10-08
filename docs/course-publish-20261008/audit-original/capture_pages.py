from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.async_api import async_playwright
from datetime import datetime,timezone
import asyncio,json,threading,hashlib
B=Path(__file__).resolve().parent
M=json.loads((B/'manifest.json').read_text());site=Path('/workspace/tiny-perceptron-vlm/outputs/tutorial-audit-20261008/site')
class Quiet(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(site)));threading.Thread(target=server.serve_forever,daemon=True).start();base='http://127.0.0.1:'+str(server.server_port)+'/'
selected={'1.3','2.1','3.4','3.7','4.2','4.7','7.4','7.22','10.2','10.5','11.17','12.5','13.14','14.1','15.5','15.6','16.3','16.4','16.14','17.8','18.1','18.6','19.6','20.7','B.3','C.7','first-steps'}
(B/'page-captures').mkdir(exist_ok=True)
async def main():
 async with async_playwright() as pw:
  browser=await pw.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
  contexts=[await browser.new_context(viewport={'width':1280,'height':800},reduced_motion='reduce') for _ in range(4)]
  q=asyncio.Queue()
  log=B/'checks/page-layout-dom.jsonl'
  results=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
  seen={r['page_id'] for r in results}
  for p in M['inventory']['pages']:
   if p['page_id'] not in seen:q.put_nowait(p)
  async def worker(ctx):
   page=await ctx.new_page()
   while not q.empty():
    p=await q.get();pid=p['page_id'];views=[]
    for width,height in [(1280,800),(390,844)]:
     await page.set_viewport_size({'width':width,'height':height})
     response=await page.goto(base+pid+'.html',wait_until='domcontentloaded',timeout=30000)
     await page.evaluate('async()=>{await document.fonts.ready;if(window.MathJax?.startup?.promise)await Promise.race([window.MathJax.startup.promise,new Promise(r=>setTimeout(r,3000))]);await Promise.all([...document.images].map(i=>i.decode().catch(()=>{})))}')
     data=await page.evaluate('''()=>{const a=document.querySelector('article');return {title:a?.querySelector('h1')?.innerText,document_overflow:document.documentElement.scrollWidth>innerWidth+1,article_overflow:a?.scrollWidth>a?.clientWidth+1,images:[...a.querySelectorAll('img')].map(i=>({src:i.getAttribute('src'),loaded:i.complete&&i.naturalWidth>0,width:i.getBoundingClientRect().width,viewport_width:i.parentElement?.getBoundingClientRect().width,scroll_width:i.parentElement?.scrollWidth})),math_nodes:document.querySelectorAll('mjx-container').length}}''')
     data.update({'viewport':[width,height],'http_status':response.status,'artifacts':[]})
     if pid in selected:
      out=B/'page-captures'/f'{pid}-{width}.png';await page.screenshot(path=str(out),full_page=True,animations='disabled');data['artifacts'].append({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'kind':'whole-page-capture'})
      for i,img in enumerate(await page.locator('article img').all()):
       figure=img.locator('xpath=ancestor::figure[1]')
       target=figure if await figure.count() else img
       if not await target.is_visible():continue
       out=B/'page-captures'/f'{pid}-{width}-figure-{i}.png'
       await target.screenshot(path=str(out),animations='disabled')
       data['artifacts'].append({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'kind':'rendered-figure-element'})
     views.append(data)
    record={'page_id':pid,'source_sha256':p['source_sha256'],'html_sha256':hashlib.sha256((site/(pid+'.html')).read_bytes()).hexdigest(),'views':views}
    results.append(record)
    with (B/'checks/page-layout-dom.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    if len(results)%20==0:print('captured DOM',len(results),flush=True)
    q.task_done()
   await page.close()
  await asyncio.gather(*(worker(c) for c in contexts));await browser.close()
  (B/'checks/page-layout-summary.json').write_text(json.dumps({'captured_at':datetime.now(timezone.utc).isoformat(),'pages':len(results),'source':'new Zensical preview of frozen authored sources; runtime outputs and reading-time annotations omitted','selected_screenshot_pages':sorted(selected),'manual_views_are_recorded_separately':True,'overflow_pages':[r['page_id'] for r in results if any(v['document_overflow'] for v in r['views'])],'missing_image_pages':[r['page_id'] for r in results if any(any(not i['loaded'] for i in v['images']) for v in r['views'])]},ensure_ascii=False,indent=2)+'\n')
asyncio.run(main());server.shutdown();server.server_close()
