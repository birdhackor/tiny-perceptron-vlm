from pathlib import Path
from playwright.sync_api import sync_playwright
import json,hashlib
OUT=Path('/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-supplemental/training-course');BASE='http://127.0.0.1:8769/'
receipt={'browser':'existing .venv Playwright Chromium','executable_path':'/usr/bin/chromium','scope':'rendered page and actual clicked navigation destinations; guide openings only','clicks':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
 for dest in ['natural-v4-student.html','natural-v4-data.html','natural-v4-training.html']:
  page.goto(BASE+'training.html',wait_until='networkidle')
  a=page.get_by_role('link',name='教材',exact=True);label=a.inner_text();href=a.get_attribute('href');a.click();page.wait_for_load_state('networkidle')
  receipt['clicks'].append({'from':BASE+'training.html','label':label,'href':href,'destination':page.url})
  toggle=page.locator('label[for="__nav_3_20"]').filter(has_text='第 20 章').first
  toggle.click()
  a=page.locator(f'.md-sidebar--primary a[href$="{dest}"]')
  a.scroll_into_view_if_needed(); visible=a.is_visible(); label=a.inner_text().strip();href=a.get_attribute('href')
  page.screenshot(path=str(OUT/(dest.removesuffix('.html')+'-nav.png')))
  a.click();page.wait_for_load_state('networkidle')
  stem=dest.removesuffix('.html');html=page.content().encode();main=page.locator('main').inner_text()
  (OUT/(stem+'.browser-rendered.html')).write_bytes(html)
  (OUT/(stem+'.browser-visible.txt')).write_text(main)
  page.screenshot(path=str(OUT/(stem+'.png')))
  item={'from':BASE+'chapter-01.html','opened_menu':'第 20 章：做一位能看圖、讀字與聽問題的助理','visible_label':label,'visible_before_click':visible,'href':href,'destination':page.url,'h1':page.locator('h1').all_inner_texts(),'h2':page.locator('h2').all_inner_texts(),'opening':main[:4500],'snapshot':stem+'.browser-rendered.html','sha256':hashlib.sha256(html).hexdigest(),'screenshot':stem+'.png','nav_screenshot':stem+'-nav.png','read_scope':'landing header and opening purpose, not full guide'}
  receipt['clicks'].append(item)
 # Main article explicit links route readers directly to warmup, natural guide and 20.2.
 for key,needle in [('W1','first-steps.html#W.1'),('intro-natural-training','natural-v4-training.html'),('T1-natural-data','natural-v4-data.html'),('intro-20-2','20.2.html')]:
  page.goto(BASE+'training.html',wait_until='networkidle')
  allmatches=page.locator(f'main a[href*="{needle}"]').all()
  a=allmatches[0] if allmatches else None
  if a is None: receipt['clicks'].append({'key':key,'found':False,'needle':needle});continue
  label=a.inner_text();href=a.get_attribute('href');a.click();page.wait_for_load_state('networkidle')
  page.screenshot(path=str(OUT/(key+'.png')))
  receipt['clicks'].append({'key':key,'from':BASE+'training.html','visible_label':label,'href':href,'destination':page.url,'h1':page.locator('h1').all_inner_texts(),'screenshot':key+'.png','scope':'explicit entrance'})
 (OUT/'browser-guide-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(receipt,ensure_ascii=False,indent=2))
 browser.close()
