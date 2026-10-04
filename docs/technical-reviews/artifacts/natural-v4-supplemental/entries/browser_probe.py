from pathlib import Path
import hashlib,json,subprocess
from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[4]
REV='f8ca78bc3ffef82b0faa352c464909b76bfd8976'
record={'preview':'http://127.0.0.1:8783/','revision':REV,'source_bytes':{},'visits':[]}
for f in ['README.md','course/README.md','assets/training/README.md']:
 current=(ROOT/f).read_bytes()
 prior=subprocess.check_output(['git','show',f'{REV}:{f}'],cwd=ROOT)
 record['source_bytes'][f]={'sha256':hashlib.sha256(current).hexdigest(),'pinned_sha256':hashlib.sha256(prior).hexdigest(),'exact_equal':current==prior}
 assert current==prior
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
 def visit(route,name):
  response=page.goto('http://127.0.0.1:8783/'+route,wait_until='networkidle',timeout=30000)
  main=page.locator('article.md-content__inner')
  text=main.inner_text()
  (OUT/(name+'.visible.txt')).write_text(text)
  page.screenshot(path=str(OUT/(name+'.png')))
  record['visits'].append({'url':page.url,'status':response.status,'title':page.title(),'visible_file':name+'.visible.txt','screenshot':name+'.png','heading':main.locator('h1').inner_text()})
 visit('readme.html','readme-browser')
 page.get_by_role('link',name='閱讀路線',exact=True).last.click()
 record['visits'].append({'action':'Click 閱讀路線 from README','result_url':page.url,'visible_heading':page.locator('article h1').inner_text()})
 visit('course.html','course-browser')
 page.get_by_role('link',name='第一節：文字怎麼變成數字？',exact=True).click()
 record['visits'].append({'action':'Click 第一節 from course R.1','result_url':page.url,'visible_heading':page.locator('article h1').inner_text()})
 main=page.locator('article.md-content__inner')
 (OUT/'lesson1.visible.txt').write_text(main.inner_text())
 page.screenshot(path=str(OUT/'lesson1-browser.png'))
 record['lesson1_links']=main.locator('a').evaluate_all('(els)=>els.map(a=>({text:a.innerText,href:a.href})).filter(x=>/Colab|Notebook|ipynb/.test(x.text+x.href))')
 record['lesson1_cpu_output']=main.locator('pre').all_inner_texts()
 # Follow local notebook download in the real browser with no cloud execution.
 notebook=next(x for x in record['lesson1_links'] if x['href'].startswith('http://127.0.0.1:8783/') and x['href'].endswith('.ipynb'))
 response=page.request.get(notebook['href'])
 nb=response.json()
 record['download']={'url':notebook['href'],'status':response.status,'cells':len(nb['cells']),'code_cells':sum(c['cell_type']=='code' for c in nb['cells']),'first_code_cell':next(c['source'] for c in nb['cells'] if c['cell_type']=='code')}
 visit('training-assets.html','assets-browser')
 page.get_by_role('link',name='新版資料說明',exact=True).click()
 record['visits'].append({'action':'Click 新版資料說明 from training assets','result_url':page.url,'visible_heading':page.locator('article h1').inner_text()})
 # Actual SVG rendering and screenshot; no generated replacement.
 response=page.goto('http://127.0.0.1:8783/figures/character_ids.svg',wait_until='load')
 page.screenshot(path=str(OUT/'character_ids-render.png'))
 record['svg']={'url':page.url,'status':response.status,'sha256':hashlib.sha256((ROOT/'course/figures/character_ids.svg').read_bytes()).hexdigest(),'screenshot':'character_ids-render.png'}
 browser.close()
(OUT/'browser-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False)[:2200])
