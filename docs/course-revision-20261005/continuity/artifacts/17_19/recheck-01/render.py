from pathlib import Path
from playwright.sync_api import sync_playwright
import json,hashlib,datetime,sys
root=Path('/workspace/tiny-perceptron-vlm');out=root/'docs/course-revision-20261005/continuity/artifacts/17_19/recheck-01';pid=sys.argv[1]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
 for label,w,h in [('desktop',1280,900),('mobile',390,844)]:
  page=b.new_page(viewport={'width':w,'height':h});url=f'http://127.0.0.1:8765/{pid}.html';page.goto(url,wait_until='domcontentloaded',timeout=15000)
  imgs=page.locator('article img, .md-content img')
  for n in range(imgs.count()):
   im=imgs.nth(n);im.scroll_into_view_if_needed(timeout=15000)
   regions=['main']
   if im.bounding_box()['height']>h-90:regions.append('bottom')
   for region in regions:
    if region=='bottom':im.evaluate('(e)=>scrollTo(0,e.getBoundingClientRect().bottom+scrollY-innerHeight+40)')
    f=out/f'{pid}-{label}-{n}-{region}.png';page.screenshot(path=str(f),full_page=False,timeout=15000)
    d={'page_id':pid,'url':url,'viewport':{'width':w,'height':h},'figure_index':n,'region':region,'image_src':im.get_attribute('src'),'screenshot':str(f.relative_to(root)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (out/'pageviews.jsonl').open('a') as s:s.write(json.dumps(d,ensure_ascii=False)+'\n')
    print(json.dumps(d,ensure_ascii=False))
  page.close()
 b.close()
