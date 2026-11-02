"""Existing preview controls, via visible labels, no build or model computation."""
import datetime,functools,http.server,json,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path('/workspace/work/omni-book-integrated-v1');OUT=ROOT/'docs/technical-reviews/artifacts/omni-P5-readme'
class Quiet(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(ROOT/'outputs/omni-P5-preview-v9')))
threading.Thread(target=server.serve_forever,daemon=True).start()
p={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_readme','command':'python docs/technical-reviews/artifacts/omni-P5-readme/probe_ui_features.py','scope':'Existing preview navigation/theme/figure/search UI only; no new build, publication, model inference.','earlier_probe_failure':'An earlier stdin probe tried to click a deliberately hidden palette radio with Locator.check(force=True), raising Element is outside viewport before UI state changed. This is corrected here by clicking its visible label; no product defect inferred from that automation misuse.'}
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path='/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox'])
  page=browser.new_page(viewport={'width':1440,'height':1000});page.goto(f'http://127.0.0.1:{server.server_address[1]}/readme.html',wait_until='networkidle')
  p['navigation_links']=page.locator('[data-md-component="navigation"] a').count();p['toc_links']=page.locator('[data-md-component="toc"] a').count();p['copy_buttons']=page.locator('button.md-clipboard').count()
  for name,operation in [
   ('palette',lambda:(page.locator('label[for="__palette_1"]').first.click(),page.locator('body').get_attribute('data-md-color-scheme'))[-1]),
   ('figure_zoom',lambda:(page.locator('.diagram-zoom').first.click(),page.locator('.diagram-zoom').first.get_attribute('aria-expanded'))[-1]),
  ]:
   try:p[name]=operation()
   except Exception as e:p[name]={'error':str(e)}
  try:
   page.locator('label[for="__search"]').first.click();query=page.locator('input[data-md-component="search-query"]');query.fill('小小感知機');page.wait_for_function("document.querySelectorAll('.md-search-result__item').length>0",timeout=8000);p['search_results']=page.locator('.md-search-result__item').count();p['search_meta']=page.locator('.md-search-result__meta').inner_text();p['first_result_titles']=page.locator('.md-search-result__title').all_inner_texts()[:3]
  except Exception as e:p['search_error']=str(e)
  browser.close()
finally:server.shutdown()
with (OUT/'ui-feature-probe-v1.json').open('x') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(p,ensure_ascii=False,indent=2))
