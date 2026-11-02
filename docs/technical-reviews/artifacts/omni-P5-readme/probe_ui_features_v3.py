"""Correct current Zensical selectors and capture actual search/copy behavior."""
import datetime,functools,http.server,json,threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path('/workspace/work/omni-book-integrated-v1');OUT=ROOT/'docs/technical-reviews/artifacts/omni-P5-readme'
class H(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
s=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(H,directory=str(ROOT/'outputs/omni-P5-preview-v9')));threading.Thread(target=s.serve_forever,daemon=True).start()
p={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer_task':'/root/omni_p5_technical_readme','command':'python docs/technical-reviews/artifacts/omni-P5-readme/probe_ui_features_v3.py','scope':'Actual existing local preview, corrected current Zensical selectors; no code/source changes, build or neural execution.'}
try:
 with sync_playwright() as pw:
  b=pw.chromium.launch(executable_path='/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox']);ctx=b.new_context(viewport={'width':1440,'height':1000},permissions=['clipboard-read','clipboard-write']);page=ctx.new_page();page.set_default_timeout(7000);page.goto(f'http://127.0.0.1:{s.server_address[1]}/readme.html',wait_until='networkidle')
  p['navigation_links']=page.locator('nav.md-nav--primary a').count();p['toc_links']=page.locator('nav.md-nav--secondary a').count();p['copy_buttons']=page.locator('button.md-code__button[data-md-type="copy"]').count()
  try:
   code=page.locator('article pre code').first.inner_text();page.locator('button.md-code__button[data-md-type="copy"]').first.click();clip=page.evaluate('navigator.clipboard.readText()');expected=(ROOT/'README.md').read_text().split('```bash\n',1)[1].split('```',1)[0];p['clipboard_matches_markdown_fence']=clip==expected;p['clipboard_matches_stripped_code']=clip.strip()==code.strip();p['clipboard_length']=len(clip);p['expected_fence_length']=len(expected);p['clipboard_tail']=repr(clip[-45:]);p['expected_fence_tail']=repr(expected[-45:])
  except Exception as e:p['copy_error']=str(e)
  page.locator('label[for="__palette_1"]').first.click();p['palette']=page.locator('body').get_attribute('data-md-color-scheme');z=page.locator('.diagram-zoom').first;z.click();p['zoom_expanded']=z.get_attribute('aria-expanded');z.click();p['zoom_shrunk']=z.get_attribute('aria-expanded')
  try:
   page.locator('button.md-search__button').click();p['visible_inputs_after_search_button']=page.locator('input').evaluate_all('(xs)=>xs.filter(x=>x.getBoundingClientRect().width>0).map(x=>({type:x.type,cls:x.className,placeholder:x.placeholder,label:x.getAttribute("aria-label"),component:x.getAttribute("data-md-component")}))')
   search=page.locator('input[placeholder="Search"]').first;search.fill('safetensors');page.wait_for_timeout(2000);p['search_dialog_text']=page.locator('dialog').all_inner_texts();p['visible_query_value']=search.input_value();p['search_body_tail']=page.locator('body').inner_text()[-4500:];p['search_input_ancestor_text']=search.evaluate('(x)=>x.parentElement.parentElement.parentElement.innerText.slice(-4500)');p['search_input_ancestor_html']=search.evaluate('(x)=>x.parentElement.parentElement.parentElement.outerHTML.slice(-4500)')
  except Exception as e:p['search_error']=str(e)
  b.close()
finally:s.shutdown()
with (OUT/'ui-feature-probe-v3.json').open('x') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(p,ensure_ascii=False,indent=2))
