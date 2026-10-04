from pathlib import Path
import hashlib,json,sys
from playwright.sync_api import sync_playwright,TimeoutError as BrowserTimeout
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[5]
REV='045ca23f28921d0feb81734390228a6e3eb97ca6';BASE='http://127.0.0.1:8785/'
record={'preview':BASE,'literal_revision':REV,'environment':{'python':sys.version,'browser_executable':'/usr/bin/chromium','device':'headless CPU'},'source_bytes':{},'visits':[],'limits':'Actual specific-page Chromium navigation and current SVG rendering. English is not presumed exported; actual English link behavior is recorded. No search-index JSON, other review report or cloud Colab execution.'}
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 context=b.new_context(viewport={'width':1440,'height':1100},permissions=['clipboard-read','clipboard-write']);page=context.new_page()
 record['environment']['chromium_version']=b.version
 for f in ['README.md','README_en.md','course/README.md','assets/training/README.md']:
  data=(ROOT/f).read_bytes();url=f'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{REV}/{f}'
  source_response=page.request.get(url,timeout=30000);assert source_response.status==200
  pinned=source_response.body();assert pinned==data
  record['source_bytes'][f]={'current_sha256':hashlib.sha256(data).hexdigest(),'pinned_sha256':hashlib.sha256(pinned).hexdigest(),'exact_equal':True,'source_url':url,'status':source_response.status,'method':'Actual bounded read-only HTTP retrieval of specific canonical file at literal revision; no Git command.'}

 provenance=page.request.get(BASE+'build-info.json');info=provenance.json()
 record['build_info']={'status':provenance.status,'sha256':hashlib.sha256(provenance.body()).hexdigest(),'fields':{k:v for k,v in info.items() if k in ['revision','lessons','executed_cpu_outputs','zensical','zensical_version','reading_time_finalized','reading_time_source_sha256']}}
 assert record['build_info']['fields']['revision']==REV and record['build_info']['fields']['lessons']==266 and record['build_info']['fields']['executed_cpu_outputs'] is True
 assert hashlib.sha256((ROOT/'outputs/site-reader-executed-v4-rms-and-notebook/build-info.json').read_bytes()).hexdigest()==record['build_info']['sha256']
 (OUT/'build-info.receipt.json').write_text(json.dumps(record['build_info'],ensure_ascii=False,indent=2)+'\n')
 def visit(route,name):
  response=page.goto(BASE+route,wait_until='networkidle',timeout=30000);main=page.locator('article.md-content__inner')
  (OUT/(name+'.visible.txt')).write_text(main.inner_text());page.screenshot(path=str(OUT/(name+'.png')))
  record['visits'].append({'url':page.url,'status':response.status,'heading':main.locator('h1').inner_text(),'visible_file':name+'.visible.txt','screenshot':name+'.png'})
 visit('readme.html','readme-browser')
 main=page.locator('article.md-content__inner');assert '影片目前提供逐幀切片' not in main.inner_text()
 assert '有程式的Notebook會在第一個程式格準備專案與套件' in main.inner_text()
 record['current_bootstrap_visible']={'chinese_condition_present':True}
 english=main.get_by_role('link',name='English',exact=True);href=english.get_attribute('href')
 record['english_link']={'href':href,'target':english.get_attribute('target'),'exporter_has_english_page':False}
 if english.get_attribute('target')=='_blank':
  with page.expect_popup(timeout=20000) as popup:english.click()
  ep=popup.value
 else:english.click();ep=page
 try:ep.wait_for_load_state('domcontentloaded',timeout=20000)
 except BrowserTimeout:record['english_link']['load_timeout']=True
 record['english_link']['actual_clicked_url']=ep.url
 record['english_link']['visible_title']=ep.title()
 english_body=ep.locator('body').inner_text()
 assert 'In notebooks with code, the first code cell prepares the repository and packages.' in english_body
 record['current_bootstrap_visible']['english_condition_present']=True
 record['english_link']['visible_excerpt']=english_body[:3000]
 (OUT/'english-link.visible.txt').write_text(english_body)
 ep.screenshot(path=str(OUT/'english-link-browser.png'))
 if ep is not page:ep.close()
 visit('readme.html','readme-browser')
 page.get_by_role('link',name='閱讀路線',exact=True).last.click();page.wait_for_load_state('networkidle')
 record['visits'].append({'action':'Click README 閱讀路線','url':page.url,'heading':page.locator('article h1').inner_text()})
 visit('course.html','course-browser')
 page.get_by_role('link',name='第一節：文字怎麼變成數字？',exact=True).click();page.wait_for_load_state('networkidle')
 main=page.locator('article.md-content__inner');(OUT/'lesson1.visible.txt').write_text(main.inner_text());page.screenshot(path=str(OUT/'lesson1-browser.png'))
 record['visits'].append({'action':'Click first lesson from course','url':page.url,'heading':main.locator('h1').inner_text()})
 record['lesson_cpu_output']=main.locator('pre').all_inner_texts()
 links=main.locator('a').evaluate_all('(els)=>els.map(a=>({text:a.innerText,href:a.href})).filter(x=>/Colab|Notebook|ipynb/.test(x.text+x.href))');record['lesson_notebook_links']=links
 local=next(x for x in links if x['href'].startswith(BASE) and x['href'].endswith('.ipynb'));response=page.request.get(local['href']);nb=response.json()
 record['actual_download']={'url':local['href'],'status':response.status,'sha256':hashlib.sha256(response.body()).hexdigest(),'cells':len(nb['cells']),'first_cell_type':nb['cells'][0]['cell_type'],'first_cell_source':nb['cells'][0]['source'],'first_code_cell_index':next(i for i,c in enumerate(nb['cells']) if c['cell_type']=='code')}
 assert record['actual_download']['first_cell_type']=='markdown' and record['actual_download']['first_code_cell_index']==2
 visit('20.2.html','lesson20-2-browser')
 main=page.locator('article.md-content__inner');nblink=next(x for x in main.locator('a').evaluate_all('(els)=>els.map(a=>({text:a.innerText,href:a.href}))') if x['href'].startswith(BASE) and x['href'].endswith('.ipynb'));response=page.request.get(nblink['href']);n=response.json()
 record['actual_download20_2']={'url':nblink['href'],'status':response.status,'cells':len(n['cells']),'code_cells':sum(c['cell_type']=='code' for c in n['cells']),'sha256':hashlib.sha256(response.body()).hexdigest()};assert record['actual_download20_2']['code_cells']==0
 visit('training-assets.html','assets-browser')
 page.get_by_role('link',name='新版資料說明',exact=True).click();page.wait_for_load_state('networkidle');record['visits'].append({'action':'Click 新版資料說明','url':page.url,'heading':page.locator('article h1').inner_text()})
 visit('readme.html','readme-browser')
 viewport=page.locator('.diagram-viewport').first;before=viewport.evaluate('(e)=>({width:e.clientWidth,scrollWidth:e.scrollWidth})');page.locator('button.diagram-zoom').click();page.screenshot(path=str(OUT/'diagram-zoom.png'))
 record['zoom']={'before':before,'after':viewport.evaluate('(e)=>({width:e.clientWidth,scrollWidth:e.scrollWidth})'),'aria_expanded':page.locator('button.diagram-zoom').get_attribute('aria-expanded')}
 # Collapse the figure before testing theme; avoids capturing a horizontally scrolled subsection.
 page.locator('button.diagram-zoom').click();page.locator('label[for=__palette_1]').click();record['theme']={'scheme':page.locator('body').get_attribute('data-md-color-scheme')};page.screenshot(path=str(OUT/'dark-mode.png'));page.locator('label[for=__palette_0]').click()
 page.locator('button.md-search__button').click();page.get_by_placeholder('Search').fill('文字怎麼變成數字');page.wait_for_timeout(1000);record['search']={'query':'文字怎麼變成數字','visible_dialog':page.locator('[role=dialog]').inner_text()[:4000]};page.screenshot(path=str(OUT/'search-browser.png'));page.keyboard.press('Escape')
 copy=page.locator('button.md-code__button').first;copy.scroll_into_view_if_needed();copy.click();record['copy']={'clipboard':page.evaluate('navigator.clipboard.readText()')};assert 'git clone' in record['copy']['clipboard']
 page.emulate_media(reduced_motion='reduce');record['reduced_motion']={'matches':page.evaluate('matchMedia("(prefers-reduced-motion: reduce)").matches')}
 response=page.goto(BASE+'figures/character_ids.svg',wait_until='load');page.screenshot(path=str(OUT/'character_ids-render.png'));record['svg']={'url':page.url,'status':response.status,'current_sha256':hashlib.sha256((ROOT/'course/figures/character_ids.svg').read_bytes()).hexdigest(),'served_sha256':hashlib.sha256(response.body()).hexdigest(),'screenshot':'character_ids-render.png'};assert record['svg']['current_sha256']==record['svg']['served_sha256']
 b.close()
(OUT/'browser-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print('Actual round3 pinned browser complete',record['english_link'],record['actual_download'],record['actual_download20_2'])
