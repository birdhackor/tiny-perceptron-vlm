from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
OUT=Path('/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-supplemental/training-course')
BASE='http://127.0.0.1:8769/'
def sha(b): return hashlib.sha256(b).hexdigest()
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
 response=page.goto(BASE+'training.html',wait_until='networkidle')
 rendered=page.content().encode()
 (OUT/'training.browser-rendered.html').write_bytes(rendered)
 body=page.locator('body').inner_text()
 (OUT/'training.browser-visible.txt').write_text(body)
 links=page.locator('a').evaluate_all('(els)=>els.map(a=>({label:a.innerText.trim(),href:a.getAttribute("href"),resolved:a.href}))')
 receipt={'browser':'Chromium via existing .venv Playwright','launch_executable':'/usr/bin/chromium','base':BASE,'initial':{'requested':BASE+'training.html','url':page.url,'status':response.status,'title':page.title(),'headings':page.locator('h1,h2,h3').all_inner_texts(),'rendered_html_sha256':sha(rendered),'rendered_html':'training.browser-rendered.html','visible_text':'training.browser-visible.txt','links':links},'navigation_clicks':[],'section_views':[],'console_errors':[]}
 page.screenshot(path=str(OUT/'training-top.png'))
 receipt['initial']['screenshot']='training-top.png'
 # Actual anchor clicks use rendered text link targets on this page.
 for section in ['T.1','T.2','T.3','T.4','T.5','T.6','T.7','T.8','T.9','T.10','T.11']:
  candidates=page.locator('a').filter(has_text=section)
  anchors=[(i,x.get_attribute('href')) for i,x in enumerate(candidates.all()) if x.get_attribute('href')=='#'+section]
  if anchors:
   candidates.nth(anchors[0][0]).click()
  else:
   page.locator(f'[id="{section}"]').scroll_into_view_if_needed()
  heading=page.locator(f'[id="{section}"]').inner_text()
  shot=section.replace('.','-')+'.png'
  page.screenshot(path=str(OUT/shot))
  receipt['section_views'].append({'section':section,'url':page.url,'visible_heading':heading,'screenshot':shot,'action':'actual anchor click' if anchors else 'actual scroll to visible heading'})
 # Traverse requested side-bar guide entrances by a rendered link click.
 for dest in ['natural-v4-student.html','natural-v4-data.html','natural-v4-training.html']:
  page.goto(BASE+'training.html',wait_until='networkidle')
  target=page.locator('a').filter(has=page.locator(':scope')) if False else page.locator(f'a[href="{dest}"]')
  count=target.count()
  if count<1:
   target=page.locator('a').filter(has_text='__unmatched__')
   for a in page.locator('a').all():
    if (a.get_attribute('href') or '').split('#')[0].endswith(dest): target=a; break
  label=target.first.inner_text().strip()
  visible=target.first.is_visible()
  href=target.first.get_attribute('href')
  target.first.click()
  page.wait_for_load_state('networkidle')
  stem=dest.removesuffix('.html')
  html=page.content().encode()
  text=page.locator('body').inner_text()
  (OUT/(stem+'.browser-rendered.html')).write_bytes(html)
  (OUT/(stem+'.browser-visible.txt')).write_text(text)
  page.screenshot(path=str(OUT/(stem+'.png')))
  main=page.locator('main')
  receipt['navigation_clicks'].append({'from':BASE+'training.html','visible_label':label,'href':href,'visible_before_click':visible,'destination':page.url,'title':page.title(),'headings':page.locator('h1,h2,h3').all_inner_texts(),'landing_text':main.inner_text()[:8000] if main.count() else text[:8000],'snapshot':stem+'.browser-rendered.html','sha256':sha(html),'visible_text':stem+'.browser-visible.txt','screenshot':stem+'.png','scope':'landing and opening purpose, not whole guide read'})
 # Check intro explicit W.1 and natural route links personally via actual rendered anchor.
 for key,needle in [('warmup-W1','first-steps.html#W.1'),('natural-TRAINING','natural-v4-training.html'),('natural-DATA','natural-v4-data.html'),('chapter20-2','20.html#20.2')]:
  page.goto(BASE+'training.html',wait_until='networkidle')
  matches=[a for a in page.locator('main a').all() if needle in (a.get_attribute('href') or '')]
  if not matches: matches=[a for a in page.locator('a').all() if needle in (a.get_attribute('href') or '')]
  if not matches:
   receipt['navigation_clicks'].append({'key':key,'needle':needle,'found':False}); continue
  a=matches[0];label=a.inner_text().strip();href=a.get_attribute('href');a.click();page.wait_for_load_state('networkidle')
  page.screenshot(path=str(OUT/(key+'.png')))
  receipt['navigation_clicks'].append({'key':key,'from':BASE+'training.html','visible_label':label,'href':href,'destination':page.url,'title':page.title(),'h1':page.locator('h1').all_inner_texts(),'screenshot':key+'.png','scope':'explicit prerequisite entrance only'})
 (OUT/'browser-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(receipt,ensure_ascii=False,indent=2))
 browser.close()
