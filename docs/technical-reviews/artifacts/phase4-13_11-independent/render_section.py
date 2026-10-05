import json
import platform
import subprocess
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

BASE=Path(__file__).resolve().parent
body=(BASE/'inputs/section.md').read_text()
html='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>13.11 independent source render</title>
<style>body {margin:0;background:#fafafa;color:#222;font-family:"Noto Sans CJK TC","Noto Sans CJK SC",sans-serif;line-height:1.65;font-size:18px} article {max-width:900px;margin:24px auto;padding:0 24px} h2{font-size:27px;line-height:1.4} pre{font-size:14px;background:#eee;padding:12px;overflow:auto;line-height:1.5} code{font-family:monospace} p{margin:16px 0} @media(max-width:450px){body{font-size:16px}article{padding:0 16px;margin:16px auto}h2{font-size:23px}}</style><article>'''+markdown.markdown(body,extensions=['fenced_code'])+'</article></html>'
(BASE/'section-render.html').write_text(html)
result={'command':'/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/phase4-13_11-independent/render_section.py',
 'environment':{'platform':platform.platform(),'renderer':'Playwright Chromium /usr/bin/chromium',
 'chromium_version':subprocess.run(['/usr/bin/chromium','--version'],capture_output=True,text=True,timeout=10).stdout.strip()},
 'source':'inputs/section.md','external_network':'none; saved self-contained HTML loaded with page.set_content',
 'scope':'Review of the exact section text/code presentation; no SVG is referenced. This is an isolated Markdown render, not a whole-course integration check.',
 'viewports':[]}
try:
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'],timeout=15000)
  for label,w,h in [('desktop',1280,800),('mobile',390,844)]:
   page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
   page.set_content(html,wait_until='load',timeout=15000)
   page.screenshot(path=str(BASE/(label+'.png')),full_page=True,timeout=15000)
   result['viewports'].append({'label':label,'width':w,'height':h,'screenshot':label+'.png',
    'document_width':page.evaluate('document.documentElement.scrollWidth'),
    'document_height':page.evaluate('document.documentElement.scrollHeight')})
   page.close()
  browser.close()
 result['status']='rendered'
except Exception as exc:
 result['status']='failed'
 result['error_type']=type(exc).__name__
 result['error']=str(exc)
(BASE/'render-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
