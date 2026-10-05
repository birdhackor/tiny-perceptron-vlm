"""Render only the reviewed raw section; no course-wide build or network."""
import hashlib
import json
import platform
import sys
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

base=Path(__file__).resolve().parents[1]
raw=(base/'inputs/section-9.2.md').read_bytes()
body=markdown.markdown(raw.decode('utf-8'),extensions=['fenced_code','tables'])
body=body.replace('<details>','<details open>')
page_html='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><style>body{margin:0;color:#18212c;font:18px/1.7 sans-serif;background:#fff}main{max-width:860px;margin:auto;padding:24px}pre{font:15px/1.5 monospace;overflow:auto;background:#f3f5f8;padding:14px}code{font-size:.9em}summary{font-weight:bold}a{color:#165fbd}h2{font-size:28px}</style><main>'+body+'</main></html>'
(base/'visual/section-only.html').write_text(page_html,encoding='utf-8')
receipt={'scope':'Section-only factual material rendering. There are no referenced figures, axes or image labels in 9.2. This is not a full course responsive-layout review. Details opened for supplemental factual inspection.','source_sha256':hashlib.sha256(raw).hexdigest(),'python':sys.version,'platform':platform.platform(),'markdown_version':markdown.__version__,'screenshots':[]}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
    receipt['chromium_version']=browser.version
    for name,width,height in [('desktop',1280,800),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        page.route('**/*',lambda route:route.abort())
        page.set_content(page_html,wait_until='load')
        page.screenshot(path=str(base/f'visual/{name}.png'),full_page=True)
        receipt['screenshots'].append({'name':name,'viewport':[width,height],'body_width':page.locator('body').evaluate('(e)=>e.scrollWidth'),'main_width':page.locator('main').evaluate('(e)=>e.getBoundingClientRect().width'),'png_sha256':hashlib.sha256((base/f'visual/{name}.png').read_bytes()).hexdigest()})
        page.close()
    browser.close()
(base/'visual/render-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
