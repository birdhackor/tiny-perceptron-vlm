import hashlib,json,sys
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path('/workspace/tiny-perceptron-vlm');scratch=Path('/workspace/work/tokenizer-dynamic-20261027/technical-course')
shots=[]; checks=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 for label,width,height in [('desktop',1440,1000),('mobile',390,844)]:
  page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
  response=page.goto('http://127.0.0.1:8000/course.html',wait_until='domcontentloaded');page.wait_for_timeout(500)
  assert response.status==200
  assert page.locator('#R\\.1').count()==1
  assert page.locator('#R\\.4').count()==1
  main=page.locator('article').first
  checks.append({'viewport':label,'status':response.status,'main_heading':main.locator('h1').inner_text(),'article_width':main.bounding_box()['width'],'viewport_width':width,'horizontal_overflow':page.evaluate('document.documentElement.scrollWidth > window.innerWidth')})
  out=scratch/f'guide-{label}-top.png';page.screenshot(path=str(out));shots.append({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'viewport':[width,height]})
  page.locator('#R\\.4').scroll_into_view_if_needed();page.wait_for_timeout(100)
  out=scratch/f'guide-{label}-R4.png';page.screenshot(path=str(out));shots.append({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'viewport':[width,height]})
  page.close()
 page=browser.new_page();response=page.goto('http://127.0.0.1:8000/1.1.html',wait_until='domcontentloaded');assert response.status==200
 assert page.locator('article').get_by_text('貓看狗，狗看貓。',exact=False).count()>0
 checks.append({'entry':'1.1.html','status':response.status,'cpu_outputs':page.locator('.output').count(),'notebook_links':page.locator('a[download]').count()})
 browser.close()
record={'command':'python /workspace/work/tokenizer-dynamic-20261027/technical-course/browser-inspect.py','environment':{'python':sys.version.split()[0],'browser':'/usr/bin/chromium','device':'desktop 1440x1000 and mobile 390x844'},'checks':checks,'screenshots':shots,'scope':'Local guide layout and first lesson entry only; no Colab login or cloud execution.'}
(root/'docs/technical-reviews/artifacts/tokenizer-20261027/course/browser-inspection.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
