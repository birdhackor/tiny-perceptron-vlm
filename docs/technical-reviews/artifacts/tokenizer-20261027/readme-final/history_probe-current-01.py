import hashlib,json,urllib.request,platform,re
from pathlib import Path
from datetime import datetime,UTC
from playwright.sync_api import sync_playwright
from scripts.stage_mathjax import validate_manifest,verify
r=Path.cwd(); s=Path('/workspace/work/tokenizer-dynamic-20261027/readme-final'); a=r/'docs/technical-reviews/artifacts/tokenizer-20261027/readme-final'
h=lambda raw:hashlib.sha256(raw).hexdigest(); m=validate_manifest(json.loads((r/'course/web/mathjax-assets.json').read_bytes())); rows=[]
for item in m['files']:
 p=r/'outputs/tokenizer-20261027/javascripts/mathjax'/item['path']; raw=p.read_bytes(); verify(raw,item)
 url='http://127.0.0.1:8000/javascripts/mathjax/'+item['path']
 with urllib.request.urlopen(url) as response: http=response.read(); status=response.status
 verify(http,item); rows.append({'path':item['path'],'sha256':h(raw),'bytes':len(raw),'status':status,'served_sha256':h(http)})
assert len(rows)==94
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox']); page=b.new_page(viewport={'width':1280,'height':900}); requests=[];blocked=[];errors=[]
 def route(route):
  url=route.request.url
  if url.startswith('http://127.0.0.1:8000/') or url.startswith('data:'):route.continue_()
  else:blocked.append(url);route.abort()
 page.route('**/*',route); page.on('request',lambda req:requests.append(req.url));page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto('http://127.0.0.1:8000/9.15.html');page.wait_for_function('window.MathJax && !!window.MathJax.typesetPromise');page.evaluate('MathJax.startup.promise');page.wait_for_load_state('networkidle')
 count=page.locator('mjx-container').count(); assert count>0
 formula=page.locator('mjx-container').first; formula.screenshot(path=str(s/'mathjax-formula-current.png'))
 readme=page.goto('http://127.0.0.1:8000/readme.html');page.wait_for_load_state('networkidle'); text=page.locator('article').inner_text(); assert '公式由網站隨附的 MathJax 在瀏覽器排版' in text
 page.goto('http://127.0.0.1:8000/1.1.html');page.wait_for_load_state('networkidle'); count_outputs=page.locator('article div.output').count(); output_text=page.locator('article div.output').all_text_contents(); assert count_outputs>0
 browser={'formula_page':'9.15.html','formula_count':count,'blocked_remote_requests':blocked,'pageerrors':errors,'requests':requests,'formula_screenshot':str(s/'mathjax-formula-current.png'),'formula_screenshot_sha256':h((s/'mathjax-formula-current.png').read_bytes()),'readme_mathjax_sentence':True,'first_lesson_outputs':count_outputs,'first_lesson_output_text':output_text,'note':'Original browser selector .executed-output returned 0 because actual class is div.output; new correct-selector observation confirms CPU outputs. No content change.'}; b.close()
receipt={'created_at':datetime.now(UTC).isoformat(),'manifest_sha256':h((r/'course/web/mathjax-assets.json').read_bytes()),'version':m['version'],'revision':m['revision'],'asset_count':len(rows),'assets':rows,'browser':browser,'probe_notes':'First attempt used1.7, which has no arithmatex formula; assertion0 diagnosed from generated HTML. Corrected representative page9.15 without editing content.','scope':'New own local 94-file byte/SHA and served-response verification; browser remote requests blocked and local formulas typeset. Not a full-site mathematical content review.'}
(a/'mathjax-current-01.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'asset_count':len(rows),'formula_count':count,'blocked_remote_requests':blocked,'pageerrors':errors,'first_lesson_outputs':count_outputs},ensure_ascii=False))
