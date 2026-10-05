from pathlib import Path
from datetime import UTC,datetime
import re,json,hashlib
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
r=Path(__file__).resolve().parent
section=(r/'inputs/section.md').read_text()
source_main=section.split('<details>')[0]
without_fence=re.sub(r'```python\n.*?```\n','',source_main,flags=re.S)
expected_paragraphs=[p.strip().replace('`','') for p in without_fence.split('\n\n') if p.strip() and not p.startswith('##')]
expected_title=section.splitlines()[0].removeprefix('## ')
expected_code=(r/'inputs/fence-1.py').read_text()
preview=BeautifulSoup((r/'preview-original.html').read_bytes(),'html.parser')
article=preview.select_one('article.md-content__inner')
paragraphs=[p.get_text() for p in article.select('p')[:len(expected_paragraphs)]]
assert paragraphs==expected_paragraphs
assert article.select_one('pre code').get_text()==expected_code
assert article.h1.get_text().removesuffix('¶')==expected_title
receipts={'checked_at':datetime.now(UTC).isoformat(),'url':'http://127.0.0.1:8765/11.13.html','source_sha256':hashlib.sha256((r/'inputs/section.md').read_bytes()).hexdigest(),'html_sha256':hashlib.sha256((r/'preview-original.html').read_bytes()).hexdigest(),'main_paragraphs_equal':True,'original_code_equal':True,'title_equal':True,'checked_paragraphs':paragraphs,'figure_count_in_source':0,'views':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 receipts['browser_version']=browser.version
 for name,width,height in [('desktop',1280,800),('mobile',390,844)]:
  page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
  page.goto(receipts['url'],wait_until='networkidle',timeout=30000)
  assert page.locator('article pre code').inner_text()==expected_code
  page.screenshot(path=str(r/f'{name}-viewport.png'))
  page.screenshot(path=str(r/f'{name}-full.png'),full_page=True)
  receipts['views'].append({'name':name,'viewport':{'width':width,'height':height},'document':page.evaluate('({width:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight})'),'article_images':page.locator('article img').count(),'code_scroll':page.locator('article pre').evaluate('(x)=>({clientWidth:x.clientWidth,scrollWidth:x.scrollWidth})'),'screenshots':[f'{name}-viewport.png',f'{name}-full.png']})
  page.close()
 browser.close()
(r/'render-receipt.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipts.items() if k!='checked_paragraphs'},ensure_ascii=False,indent=2))
