from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
OUT=Path('/workspace/tiny-perceptron-vlm/docs/reader-reviews/artifacts/natural-v4-supplemental/training-course')
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
 page.set_default_timeout(4000)
 response=page.goto('http://127.0.0.1:8769/training.html',wait_until='networkidle')
 article=page.locator('article');body=article.inner_text();(OUT/'training.browser-article-complete.txt').write_text(body)
 rendered=page.content().encode();(OUT/'training.browser-final-rendered.html').write_bytes(rendered)
 receipt={'url':page.url,'status':response.status,'executable_path':'/usr/bin/chromium','viewport':{'width':1440,'height':1100},'rendered_html_sha256':hashlib.sha256(rendered).hexdigest(),'visible_text_sha256':hashlib.sha256(body.encode()).hexdigest(),'h1':page.locator('h1').all_inner_texts(),'article_height':article.bounding_box()['height'],'paragraphs':article.locator('p').count(),'tables':article.locator('table').count(),'code_blocks':article.locator('pre').count(),'direct_svg_images':article.locator('img[src$=".svg"]').count(),'section_clicks':[],'tiles':[],'tables_view':[]}
 for section in ['T.1','T.2','T.3','T.4','T.5','T.6','T.7','T.8','T.9','T.10','T.11']:
  link=page.locator(f'a[href="#{section}"]').first
  label=link.inner_text().strip();link.click()
  heading=page.locator(f'[id="{section}"]');shot=section.replace('.','-')+'-browser-final.png'
  page.screenshot(path=str(OUT/shot))
  receipt['section_clicks'].append({'section_id':section,'visible_label':label,'destination':page.url,'heading':heading.inner_text(),'screenshot':shot})
  (OUT/'browser-training-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 # Full actual rendered article clips in its DOM order, including every paragraph/table/command.
 page.evaluate('window.scrollTo(0,0)');bounds=article.bounding_box();height=bounds['height'];tile_h=2600
 for i,off in enumerate(range(0,int(height)+1,tile_h),1):
  h=min(tile_h,height-off)
  if h<=0:break
  shot=f'training-article-tile-{i:02}.png'
  page.screenshot(path=str(OUT/shot),clip={'x':bounds['x'],'y':bounds['y']+off,'width':bounds['width'],'height':h},capture_beyond_viewport=True)
  receipt['tiles'].append({'order':i,'from_article_y':off,'height':h,'screenshot':shot})
 for i,table in enumerate(article.locator('table').all(),1):
  name=f'training-table-{i:02}.png';table.screenshot(path=str(OUT/name))
  receipt['tables_view'].append({'order':i,'screenshot':name,'headers':table.locator('th').all_inner_texts()})
 (OUT/'browser-training-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(receipt,ensure_ascii=False,indent=2))
 b.close()
