"""Render the current raw section with the repository diagram wrapper and course CSS."""
from pathlib import Path
import sys, re, json, threading, functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import markdown
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from scripts.export_course import diagram_html
ART=Path(__file__).resolve().parents[1]
DEST=ART/'render'
body=(ART/'original-execution/section.md').read_text()
body=re.sub(r'!\[([^\]]*)\]\(\.\./figures/([^\)]+)\)',lambda m:diagram_html('../inputs/course/figures/'+m[2],m[1],'review-10-7-diagram'),body)
html=markdown.markdown(body,extensions=['fenced_code','tables'])
page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="../inputs/course/web/course.css"><style>body{margin:0;background:#fff;font-family:"Noto Sans CJK TC",sans-serif;color:#17293c}main{max-width:860px;margin:0 auto;padding:24px}p{line-height:1.8}pre{overflow:auto;padding:16px;background:#f4f6f8}code{font-size:14px}img{max-width:100%}h2{font-size:26px} @media(max-width:500px){main{padding:16px}h2{font-size:24px}}</style><main class="md-typeset">'''+html+'</main></html>'
(DEST/'section.html').write_text(page)
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(ART)))
threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu'])
    records=[]
    for label,width,height in [('desktop',1280,800),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        page.goto(f'http://127.0.0.1:{server.server_port}/render/section.html',wait_until='load')
        page.screenshot(path=str(DEST/f'{label}-section.png'),full_page=True)
        image=page.locator('img.diagram')
        image.screenshot(path=str(DEST/f'{label}-figure.png'))
        records.append(dict(viewport=[width,height],image_bbox=image.bounding_box(),
                         image_complete=image.evaluate('(img)=>img.complete && img.naturalWidth>0'),
                         document_width=page.evaluate('document.documentElement.scrollWidth'),
                         source='Raw 10.7 section, repository diagram_html wrapper and course.css; local excerpt layout, not the full site navigation.'))
        page.close()
    version=browser.version
    browser.close()
server.shutdown()
server.server_close()
print(json.dumps(dict(chromium=version,markdown=markdown.__version__,records=records),indent=2))
