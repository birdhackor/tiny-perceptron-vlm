from pathlib import Path
import json,re,hashlib
from playwright.sync_api import sync_playwright
root=Path.cwd(); out=root/'docs/technical-reviews/artifacts/natural-v4-supplemental/site-validation-publishing'; base='http://127.0.0.1:8788/'
def norm(s): return re.sub(r'\s+','',s)
def pieces(source):
 fence=False
 for line in source.splitlines():
  if line.startswith('```'): fence=not fence; continue
  if not line.strip() or re.fullmatch(r'[| :\-]+',line): continue
  if fence: yield line; continue
  line=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',line)
  line=re.sub(r'^#{1,6}\s+|^\d+\.\s+','',line).replace('`','')
  if line.startswith('|'):
   yield from [x.strip() for x in line.strip('|').split('|') if x.strip()]
  else: yield line
receipt={'events':[], 'source_correspondence':[], 'screenshots':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox']); context=browser.new_context(viewport={'width':1365,'height':900},accept_downloads=True); page=context.new_page()
 for name in ['validation','publishing']:
  response=page.goto(base+name+'.html',wait_until='networkidle'); article=page.locator('article.md-content__inner'); text=article.inner_text(); (out/(name+'-visible.txt')).write_text(text)
  png=out/(name+'-page.png'); page.screenshot(path=str(png),full_page=True); receipt['screenshots'].append({'url':page.url,'path':str(png.relative_to(root)),'sha256':hashlib.sha256(png.read_bytes()).hexdigest()})
  checks=list(pieces((root/f'docs/{name}.md').read_text())); missing=[s for s in checks if norm(s) not in norm(text)]
  receipt['source_correspondence'].append({'source':f'docs/{name}.md','url':page.url,'http':response.status,'compared_author_lines_or_table_cells':len(checks),'missing':missing,'method':'Each substantive prose line, heading, table cell, and command line, with Markdown link/backtick syntax removed and whitespace normalized, must occur in personally rendered article text.'})
  disk=root/'outputs/site-reader-executed-v4-publishing-prerequisite'/f'{name}.html'; raw=response.body(); assert raw==disk.read_bytes(); receipt['events'].append({'url':page.url,'http_html_sha256':hashlib.sha256(raw).hexdigest(),'served_disk_byte_equal':True,'generator':page.locator('meta[name=generator]').get_attribute('content')})
 # Use assigned page links in browser, then inspect actual lesson actions.
 page.get_by_role('link',name='實作驗證',exact=True).click(); page.wait_for_url('**/validation.html'); receipt['events'].append({'click':'publishing body 實作驗證','arrived_url':page.url,'visible_title':page.locator('article h1').inner_text()})
 page.get_by_role('link',name='教材發布',exact=True).click(); page.wait_for_url('**/publishing.html'); receipt['events'].append({'click':'validation body 教材發布','arrived_url':page.url})
 page.locator('article').get_by_role('link',name='W.1',exact=True).click(); page.wait_for_url('**/first-steps.html#W.1'); receipt['events'].append({'click':'publishing body W.1','arrived_url':page.url,'anchor_exists':page.locator('[id="W.1"]').count()==1})
 page.goto(base+'1.1.html',wait_until='networkidle'); receipt['events'].append({'representative_lesson':page.url,'visible_output_count':page.locator('.output').count(),'colab_href':page.locator('.actions a').first.get_attribute('href')})
 with page.expect_download() as info: page.get_by_role('link',name='下載本節 .ipynb',exact=True).click()
 d=info.value; target=out/'download-1.1.ipynb'; d.save_as(str(target)); downloaded=json.loads(target.read_text()); canonical=json.loads((root/'notebooks/01/1.1.ipynb').read_text()); assert downloaded==canonical
 receipt['events'].append({'click':'下載本節 .ipynb','suggested_filename':d.suggested_filename,'download_url':d.url,'canonical_json_equal':True,'download_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'first_code_is_setup':next(c for c in downloaded['cells'] if c['cell_type']=='code')['metadata'].get('course_setup') is True})
 png=out/'lesson-download-actions.png'; page.screenshot(path=str(png)); receipt['screenshots'].append({'url':page.url,'path':str(png.relative_to(root)),'sha256':hashlib.sha256(png.read_bytes()).hexdigest()})
 # The ASR explanation's necessary prerequisite diagram; render original served SVG.
 page.goto(base+'figures/natural-v4-asr-two-routes.svg',wait_until='networkidle'); png=out/'asr-two-routes-render.png'; page.screenshot(path=str(png)); receipt['screenshots'].append({'url':page.url,'path':str(png.relative_to(root)),'sha256':hashlib.sha256(png.read_bytes()).hexdigest()})
 receipt['svg_original_sha256']=hashlib.sha256((root/'course/figures/natural-v4-asr-two-routes.svg').read_bytes()).hexdigest()
 receipt['browser_version']=browser.version; browser.close()
(out/'browser-receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(receipt,indent=2,ensure_ascii=False))
