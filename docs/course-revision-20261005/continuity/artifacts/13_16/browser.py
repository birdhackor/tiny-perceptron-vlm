import sys,json,hashlib,pathlib,datetime
from playwright.sync_api import sync_playwright
BASE=pathlib.Path('/workspace/tiny-perceptron-vlm/docs/course-revision-20261005/continuity/artifacts/13_16')
pageid=sys.argv[1]; shot='--shots' in sys.argv
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
 for width,height,label in [(1280,900,'desktop'),(390,844,'mobile')]:
  page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
  url='http://127.0.0.1:8765/'+pageid+'.html'
  resp=page.goto(url,wait_until='domcontentloaded',timeout=15000)
  page.evaluate("document.querySelectorAll('main details').forEach(d=>d.open=true)")
  images=page.locator('main img')
  paths=[]
  if shot:
   for i in range(images.count()):
    el=images.nth(i); el.scroll_into_view_if_needed(timeout=15000)
    box=el.bounding_box(); top=box['y']+page.evaluate('window.scrollY')
    for segment,offset in enumerate(range(0,max(1,int(box['height'])),height-180)):
     page.evaluate('(y)=>window.scrollTo(0,Math.max(0,y-120))',top+offset)
     out=BASE/f'{pageid}-{label}-{i}-s{segment}.png';page.screenshot(path=str(out),full_page=False,timeout=15000)
     paths.append({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'image_src':el.get_attribute('src'),'image_alt':el.get_attribute('alt'),'segment':segment})
  entry={'visited_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'page_id':pageid,'url':url,'viewport':{'width':width,'height':height},'status':resp.status,'screenshots':paths}
  with (BASE/'pageviews.jsonl').open('a') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
  print(json.dumps(entry,ensure_ascii=False))
  page.close()
 browser.close()
