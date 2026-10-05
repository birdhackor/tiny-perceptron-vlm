import json,time,hashlib,urllib.request,platform,shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
R=Path.cwd();A=R/'docs/technical-reviews/artifacts/phase4-13_15-independent';meta=json.loads((R/(A/'context-reinspection/latest-start-path.txt').read_text().strip()).read_text());O=R/meta['reinspection_path'];url=meta['preview_url'];result={'url':url,'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-13_15-independent/context-reinspection/render-current-preview.py','scope':'Actual current exported 13.15 page, desktop/mobile view; this is a factual context reinspection, not a new full reader acceptance review','python':platform.python_version(),'screenshots':[]};started=time.monotonic()
try:
 raw=urllib.request.urlopen(url,timeout=15).read();(O/'13.15-preview.html').write_bytes(raw);result['html_sha256']=hashlib.sha256(raw).hexdigest()
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'],timeout=15000);result['chromium_version']=browser.version
  for label,w,h in [('desktop',1280,800),('mobile',390,844)]:
   page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1);page.goto(url,wait_until='domcontentloaded',timeout=15000);page.screenshot(path=str(O/(label+'-current-top.png')))
   result['screenshots'].append({'file':label+'-current-top.png','viewport':[w,h]})
   details=page.locator('details')
   if details.count():
    details.evaluate_all('(els)=>els.forEach(e=>e.open=true)')
    code=page.locator('pre').first
    if code.count():code.scroll_into_view_if_needed();page.screenshot(path=str(O/(label+'-current-code.png')));result['screenshots'].append({'file':label+'-current-code.png','viewport':[w,h]})
   text=page.locator('body').inner_text();assert '13.15' in text and '一輪選卡與更新' in text and '6／18' in text and '12／18' in text
   result[label+'_content_check']='Lesson heading and current 6/18,12/18 text present; screenshot produced, awaiting own visual inspection';page.close()
  browser.close();result['status']='completed'
except Exception as e:result['status']='failed';result['error']=type(e).__name__+': '+str(e)
result['elapsed_seconds']=time.monotonic()-started;(O/'preview-render-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
shutil.copyfile(Path(__file__),O/'render-current-preview-executed.py')
if result['status']!='completed':raise SystemExit(1)
