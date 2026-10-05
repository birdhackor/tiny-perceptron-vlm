import json,pathlib,datetime,hashlib
from playwright.sync_api import sync_playwright
A=pathlib.Path('/workspace/tiny-perceptron-vlm/docs/course-revision-20261005/continuity/artifacts/13_16/recheck-02')
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
 for w,h,label in [(1280,900,'desktop'),(390,844,'mobile')]:
  page=b.new_page(viewport={'width':w,'height':h},device_scale_factor=1);url='http://127.0.0.1:8765/16.4.html';resp=page.goto(url,wait_until='domcontentloaded',timeout=15000)
  page.evaluate("document.querySelectorAll('main details').forEach(d=>d.open=true)");el=page.locator('main p').filter(has_text='這份品質表使用').first;el.scroll_into_view_if_needed(timeout=15000)
  box=el.bounding_box();top=box['y']+page.evaluate('window.scrollY');page.evaluate('(y)=>window.scrollTo(0,Math.max(0,y-120))',top)
  out=A/f'16.4-{label}-task-paragraph.png';page.screenshot(path=str(out),full_page=False,timeout=15000)
  d={'visited_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'page_id':'16.4','url':url,'viewport':{'width':w,'height':h},'status':resp.status,'observed_text':el.inner_text(),'screenshots':[{'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'target':'current quality-table task paragraph and surrounding public prose','image_src':None,'image_alt':None}]}
  with (A/'pageviews.jsonl').open('a') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
  print(json.dumps(d,ensure_ascii=False));page.close()
 b.close()
