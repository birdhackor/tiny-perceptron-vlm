import json,hashlib
from pathlib import Path
from datetime import datetime,UTC
from playwright.sync_api import sync_playwright
r=Path('/workspace/tiny-perceptron-vlm'); s=Path('/workspace/work/tokenizer-dynamic-20261027/readme-final'); out=r/'docs/technical-reviews/artifacts/tokenizer-20261027/readme-final'
records=[]
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 for width,height in [(1280,900),(390,844)]:
  page=b.new_page(viewport={'width':width,'height':height},reduced_motion='reduce')
  errors=[]; page.on('pageerror',lambda e: errors.append(str(e)))
  response=page.goto('http://127.0.0.1:8000/readme.html'); page.wait_for_load_state('networkidle')
  page.screenshot(path=str(s/f'readme-{width}.png'),full_page=True)
  img=page.locator('article img.diagram').first
  img.screenshot(path=str(s/f'figure-in-page-{width}.png'))
  records.append({'url':page.url,'http_status':response.status,'viewport':[width,height],'screenshot':str(s/f'readme-{width}.png'),'screenshot_sha256':hashlib.sha256((s/f'readme-{width}.png').read_bytes()).hexdigest(),'figure_screenshot':str(s/f'figure-in-page-{width}.png'),'figure_screenshot_sha256':hashlib.sha256((s/f'figure-in-page-{width}.png').read_bytes()).hexdigest(),'pageerrors':errors,'overflow':page.evaluate('document.documentElement.scrollWidth > innerWidth'),'text_markers':{x:x in page.locator('article').inner_text() for x in ['313','339','3,734','MathJax','18–28']},'img':img.evaluate('(e)=>({src:e.src,complete:e.complete,w:e.naturalWidth,h:e.naturalHeight})')})
  page.close()
 page=b.new_page(viewport={'width':1000,'height':400},reduced_motion='reduce')
 svg=r/'course/figures/rewrite-01-character-ids.svg'
 page.set_content(svg.read_text()); page.screenshot(path=str(s/'figure-render.png'))
 records.append({'source':str(svg.relative_to(r)),'source_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'method':'page.set_content of verified SVG, reduced motion','viewport':[1000,400],'screenshot':str(s/'figure-render.png'),'screenshot_sha256':hashlib.sha256((s/'figure-render.png').read_bytes()).hexdigest()})
 page=b.new_page(); page.goto('http://127.0.0.1:8000/1.1.html'); page.wait_for_load_state('networkidle')
 records.append({'url':page.url,'cpu_outputs':page.locator('.executed-output').count(),'download_links':page.locator('a[download]').count(),'colab':page.locator('a[href*="colab.research"]').all_text_contents()})
 b.close()
(out/'browser-original-01.json').write_text(json.dumps({'created_at':datetime.now(UTC).isoformat(),'records':records,'inspection':'Screenshots generated here; visual observations added in review after view_image.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False))
