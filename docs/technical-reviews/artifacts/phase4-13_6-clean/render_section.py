import json
import sys
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

BASE=Path(__file__).resolve().parent
text=(BASE/'inputs/section.md').read_text()
html=markdown.markdown(text.replace('<details>', '<details open markdown="1">'), extensions=['fenced_code','tables','md_in_html'])
page=BASE/'section-render.html'
page.write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><style>body{font-family:"Noto Sans CJK TC",sans-serif;line-height:1.65;margin:0;background:white;color:#16202a}main{max-width:960px;margin:24px auto;padding:0 20px}pre{overflow:auto;background:#f2f4f7;padding:14px;font-size:14px}code{font-family:monospace}table{border-collapse:collapse;font-size:15px;max-width:100%}td,th{border:1px solid #b5bcc4;padding:8px;vertical-align:top}details{border:1px solid #b5bcc4;padding:12px;margin-top:20px}h2{line-height:1.3}</style><main>'+html+'</main></html>')
result={'command': '.venv/bin/python docs/technical-reviews/artifacts/phase4-13_6-clean/render_section.py', 'python':sys.version, 'markdown_version':markdown.__version__, 'html':str(page),'scope':'Frozen section-only HTML render; details opened; not production-site integration.'}
with sync_playwright() as p:
    try:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'],timeout=15000)
        result['browser_version']=browser.version
        for name,width,height in [('desktop',1280,800),('mobile',390,844)]:
            tab=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
            tab.set_content(page.read_text(),wait_until='load',timeout=15000)
            tab.screenshot(path=str(BASE/(name+'.png')),full_page=True,timeout=15000)
            result[name]={'viewport':[width,height],'table_count':tab.locator('table').count(),'screenshot':name+'.png'}
            tab.close()
        browser.close()
        result['status']='success'
    except Exception as e:
        result.update(status='failed',exception_type=type(e).__name__,error=str(e))
(BASE/'execution/render-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
