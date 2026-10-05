from pathlib import Path
import hashlib,json,re,sys,urllib.request
from datetime import datetime,UTC
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
OUT=Path(__file__).resolve().parent
URL='http://127.0.0.1:8765/11.11.html'
section=(OUT/'section.md').read_text()
expected_code=(OUT/'fence-1.py').read_text().rstrip('\n')
with urllib.request.urlopen(URL,timeout=20) as r:html=r.read()
soup=BeautifulSoup(html,'html.parser');article=soup.select_one('article.md-content__inner')
assert article is not None
rendered_code=article.select_one('pre code').get_text().rstrip('\n')
assert rendered_code==expected_code
expected=[]
prose=re.sub(r'```.*?```','',section,flags=re.S)
for p in prose.split('\n\n'):
 p=p.strip()
 if not p or p.startswith(('##','<details>','</details>','<summary>')):continue
 p=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',p)
 expected.append(p)
normalize=lambda t:re.sub(r'\s+','',t)
rendered_paragraphs=[p.get_text() for p in article.select('p')]
checks=[]
for p in expected:
 equal=[i for i,q in enumerate(rendered_paragraphs) if normalize(p)==normalize(q)]
 assert len(equal)==1,(p,equal)
 checks.append({'source_paragraph':p,'rendered_paragraph_index':equal[0],'exact_text_ignoring_whitespace':True})
(OUT/'served-11.11.html').write_bytes(html)
(OUT/'served-article.html').write_text(str(article)+'\n')
result={'accessed_at':datetime.now(UTC).isoformat(),'url':URL,'source_section_sha256':hashlib.sha256((OUT/'section.md').read_bytes()).hexdigest(),
        'served_html_sha256':hashlib.sha256(html).hexdigest(),'code_exact_match':True,'paragraph_checks':checks,'renders':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu'])
 result['chromium_version']=browser.version
 for name,width,height in [('desktop',1280,800),('mobile',390,844)]:
  context=browser.new_context(viewport={'width':width,'height':height},device_scale_factor=1)
  page=context.new_page()
  page.goto(URL,wait_until='domcontentloaded',timeout=30000)
  page.locator('article.md-content__inner').wait_for()
  assert page.locator('article.md-content__inner pre code').first.inner_text().rstrip('\n')==expected_code
  page.screenshot(path=str(OUT/f'{name}-{width}x{height}-full.png'),full_page=True,timeout=20000)
  page.screenshot(path=str(OUT/f'{name}-{width}x{height}-viewport.png'),full_page=False,timeout=20000)
  metrics=page.evaluate('''() => {const a=document.querySelector('article.md-content__inner');return {viewportWidth:innerWidth,viewportHeight:innerHeight,documentScrollWidth:document.documentElement.scrollWidth,articleWidth:a.getBoundingClientRect().width,articleScrollWidth:a.scrollWidth,preCount:a.querySelectorAll('pre').length,imageCount:a.querySelectorAll('img').length}}''')
  result['renders'].append({'name':name,'viewport':[width,height],'dom_metrics':metrics,
                           'full_page_screenshot':f'{name}-{width}x{height}-full.png',
                           'viewport_screenshot':f'{name}-{width}x{height}-viewport.png'})
  context.close()
 browser.close()
(OUT/'render-receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
