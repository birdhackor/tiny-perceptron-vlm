import sys,json,hashlib,pathlib,datetime
from playwright.sync_api import sync_playwright
BASE=pathlib.Path('/workspace/tiny-perceptron-vlm/docs/course-revision-20261005/continuity/artifacts/13_16/recheck-01')
pageid=sys.argv[1]; idx=int(sys.argv[2])
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
 for w,h,label in [(1280,900,'desktop'),(390,844,'mobile')]:
  page=b.new_page(viewport={'width':w,'height':h},device_scale_factor=1);url='http://127.0.0.1:8765/'+pageid+'.html'; resp=page.goto(url,wait_until='domcontentloaded',timeout=15000)
  page.evaluate("document.querySelectorAll('main details').forEach(d=>d.open=true)");el=page.locator('main pre').nth(idx);el.scroll_into_view_if_needed(timeout=15000)
  box=el.bounding_box();top=box['y']+page.evaluate('window.scrollY');page.evaluate('(y)=>window.scrollTo(0,Math.max(0,y-120))',top)
  out=BASE/f'{pageid}-{label}-code{idx}.png';page.screenshot(path=str(out),full_page=False,timeout=15000)
  entry={'visited_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'page_id':pageid,'url':url,'viewport':{'width':w,'height':h},'status':resp.status,'screenshots':[{'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'target':'main pre index '+str(idx),'image_src':None,'image_alt':None}],'code_text':el.inner_text(),'code_class':el.evaluate('(e)=>e.className')}
  with (BASE/'pageviews.jsonl').open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
  print(json.dumps(entry,ensure_ascii=False));page.close()
 b.close()
