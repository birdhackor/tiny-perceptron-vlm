import sys,json,datetime,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
PAGE=sys.argv[1]
BASE=Path('docs/course-revision-20261006-phase6/continuity/artifacts/a')
OUT=BASE/'browser';OUT.mkdir(exist_ok=True)
URL=f'http://127.0.0.1:8793/{PAGE}.html'
records=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-gpu'],headless=True)
 for name,w,h in [('desktop',1440,1000),('mobile',390,1000)]:
  page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
  response=page.goto(URL,wait_until='networkidle');page.evaluate('document.fonts.ready');page.locator('img').evaluate_all('(imgs)=>Promise.all(imgs.map(i=>i.decode().catch(()=>{})))')
  items=page.locator('main img, article img').all()
  if not items:items=page.locator('img').all()
  for index,img in enumerate(items):
   src=img.get_attribute('src');alt=img.get_attribute('alt');box=img.bounding_box();height=box['height'];top=box['y']
   start=max(0,top-160);end=top+height+140
   positions=[start]
   while positions[-1]+h<end:positions.append(positions[-1]+700)
   files=[]
   for part,y in enumerate(positions):
    page.evaluate('(y)=>window.scrollTo(0,y)',y);page.wait_for_timeout(180)
    path=OUT/f'{PAGE}-{name}-fig{index+1}-viewport{part+1}.png';page.screenshot(path=str(path),full_page=False);files.append(str(path))
   rect=img.bounding_box(); records.append({'page_id':PAGE,'url':URL,'http_status':response.status,'viewport':{'width':w,'height':h},'image_index':index+1,'src':src,'alt':alt,'intrinsic':img.evaluate('(i)=>({width:i.naturalWidth,height:i.naturalHeight})'),'document_box':box,'end_viewport_box':rect,'scroll_positions':positions,'viewed_files':files,'body_scroll_width':page.evaluate('document.body.scrollWidth'),'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  page.close()
 browser.close()
path=OUT/f'{PAGE}-capture.json';path.write_text(json.dumps(records,ensure_ascii=False,indent=2))
print(json.dumps(records,ensure_ascii=False,indent=2))
