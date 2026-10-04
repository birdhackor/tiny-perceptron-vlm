import json,hashlib,math
from pathlib import Path
from playwright.sync_api import sync_playwright
p=Path('docs/reader-reviews/artifacts/natural-v4-supplemental/training-course/round2')
base='http://127.0.0.1:8770/'
h=lambda b:hashlib.sha256(b).hexdigest()
receipt={'reviewer_task':'/root/v4_review_coordinator/reader_whole_training_course','round':2,'pages':[]}
with sync_playwright() as w:
 b=w.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':2700},device_scale_factor=1)
 for name in ['13.1','13.4']:
  response=page.goto(base+name+'.html',wait_until='networkidle');raw=response.body();(p/(name+'.prerequisite.http-response.html')).write_bytes(raw)
  article=page.locator('article'); txt=article.inner_text();(p/(name+'.prerequisite.rendered-visible.txt')).write_text(txt)
  entry={'page_id':name,'url':page.url,'h1':article.locator('h1').all_inner_texts(),'http_sha256':h(raw),'http_local_exact_byte_match':raw==Path('outputs/site-reader-executed-v4/'+name+'.html').read_bytes(),'visible_text_sha256':h(txt.encode()),'scope':'full necessary prerequisite article including actual root-build CPU outputs','screens':[]}
  bounds=article.bounding_box(); entry['article_height']=bounds['height']
  for n,offset in enumerate(range(0,math.ceil(bounds['height']),2400),1):
   page.evaluate('(y)=>window.scrollTo(0,y)',bounds['y']+offset-96);page.wait_for_timeout(100)
   filename=f'{name}.prerequisite-tile-{n:02}.png';page.screenshot(path=str(p/filename),clip={'x':bounds['x'],'y':96,'width':bounds['width'],'height':2600});entry['screens'].append({'order':n,'offset':offset,'screenshot':filename,'actual_scroll_y':page.evaluate('window.scrollY')})
  receipt['pages'].append(entry)
 page.goto(base+'13.1.html',wait_until='networkidle')
 nav=page.locator('.md-footer__link--prev');nav.scroll_into_view_if_needed();navtxt=nav.inner_text();navhref=nav.get_attribute('href')
 with page.expect_navigation(wait_until='networkidle') as result:nav.click()
 response=result.value;raw=response.body();filename='chapter-13';(p/(filename+'.http-response.html')).write_bytes(raw)
 txt=page.locator('article').inner_text();(p/(filename+'.rendered-visible.txt')).write_text(txt);page.screenshot(path=str(p/(filename+'.png')))
 receipt['chapter_13_intro_navigation']={'from':base+'13.1.html','visible_label':navtxt,'href':navhref,'destination':page.url,'h1':page.locator('article h1').all_inner_texts(),'http_sha256':h(raw),'http_local_exact_byte_match':raw==Path('outputs/site-reader-executed-v4/chapter-13.html').read_bytes(),'screenshot':filename+'.png','read_scope':'entire chapter route introduction; subsection links only, not full chapter'}
 b.close()
(p/'browser-prerequisite-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False,indent=2))
