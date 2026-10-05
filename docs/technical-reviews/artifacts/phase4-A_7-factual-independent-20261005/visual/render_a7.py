import hashlib
import json
import subprocess
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
source=HERE.parent/'execution/section.md'
raw=source.read_bytes()
body=markdown.markdown(raw.decode('utf-8'),extensions=['fenced_code','tables'])
body=body.replace('<details>','<details open>')
html='''<!doctype html><meta charset="utf-8"><title>A.7 frozen content render</title>
<style>body{margin:24px auto;max-width:900px;padding:0 18px;font:18px/1.7 sans-serif;color:#172234;background:white}
pre{font:14px/1.55 monospace;padding:16px;background:#eef2f5;white-space:pre-wrap;overflow-wrap:anywhere}
code{font-size:.9em}details{border:1px solid #ccd4dc;padding:12px}summary{font-weight:bold}h2{font-size:27px}</style>'''+body
(HERE/'section-render.html').write_text(html,encoding='utf-8')
results=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 for width,height,name in [(1280,800,'desktop'),(390,844,'mobile')]:
  page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
  page.set_content(html,wait_until='load')
  page.screenshot(path=str(HERE/f'{name}.png'),full_page=True)
  results.append({'viewport':[width,height],'screenshot':f'{name}.png',
   'source_sha256':hashlib.sha256(raw).hexdigest(),'image_count':page.locator('img').count(),
   'open_details':page.locator('details[open]').count(),'body_text_sha256':hashlib.sha256(page.inner_text('body').encode()).hexdigest(),
   'scope':'Actual Chromium render of frozen A.7 Markdown with details expanded; standalone content, not production theme.'})
  page.close()
 browser.close()
receipt={'browser_version':subprocess.check_output(['/usr/bin/chromium','--version'],text=True).strip(),
 'render_results':results}
(HERE/'render-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
