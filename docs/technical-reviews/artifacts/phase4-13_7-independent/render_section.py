"""Render only the frozen section, using original Markdown and course CSS."""
import hashlib
import json
import platform
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

ART=Path(__file__).resolve().parent
ROOT=ART.parents[3]
source=ART/'original-execution/section.md'
css=ROOT/'course/web/course.css'
(ART/'inputs/course.css').write_bytes(css.read_bytes())
# Same details markdown attribute conversion as scripts/export_course.py:116.
body=source.read_text().replace('<details>','<details markdown="1">')
html=markdown.markdown(body,extensions=['fenced_code','md_in_html'])
wrapper='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>13.7 frozen input render</title><style>
html{font-size:20px}body{margin:0;background:#fff;color:#222;font-family:"Noto Sans CJK TC","Noto Sans CJK SC",sans-serif}
main{max-width:860px;margin:32px auto;padding:0 24px}pre{background:#f5f5f5;padding:16px;overflow-x:auto;font-size:14px;line-height:1.6}code{font-family:monospace;font-size:.85em}details{margin:24px 0;border:1px solid #ccc;padding:12px}summary{cursor:pointer}h2{font-size:26px}
@media(max-width:500px){main{margin:16px auto;padding:0 18px}h2{font-size:23px}}
'''+css.read_text()+'</style><body><main class="md-typeset">'+html+'</main></body></html>'
(ART/'section-render.html').write_text(wrapper)
record={'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-13_7-independent/render_section.py',
        'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'course_css_sha256':hashlib.sha256(css.read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),'markdown':markdown.__version__,'browser':'/usr/bin/chromium'},
        'scope':'Isolated original section, course typography CSS and explicit responsive wrapper; not a full production-site build.',
        'screenshots':[]}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,timeout=15000,
                              args=['--no-sandbox','--disable-gpu','--disable-dev-shm-usage'])
    record['environment']['chromium_version']=browser.version
    for name,w,h in [('desktop',1280,800),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
        page.set_content(wrapper,wait_until='load',timeout=15000)
        for details_open in [False,True]:
            page.locator('details').evaluate('(e, state) => {e.open=state}',details_open)
            path=ART/f'{name}-{"expanded" if details_open else "collapsed"}.png'
            page.screenshot(path=str(path),full_page=True,timeout=15000)
            record['screenshots'].append({'path':str(path.relative_to(ROOT)),'width':w,'viewport_height':h,
                                         'details_open':details_open,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                         'document_scroll_width':page.evaluate('document.documentElement.scrollWidth'),
                                         'client_width':page.evaluate('document.documentElement.clientWidth')})
        page.close()
    browser.close()
(ART/'render-measurements.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
