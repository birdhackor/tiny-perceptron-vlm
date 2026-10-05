"""Render only frozen 13.10 in Chromium at desktop/mobile widths."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import markdown

A=Path(__file__).resolve().parents[1]
ROOT=A.parents[3]
css=(ROOT/'course/web/course.css').read_text()
(A/'render/course.css').write_text(css)
body=markdown.markdown((A/'inputs/section-13.10.md').read_text(),extensions=['tables','fenced_code'])
base='''html { font-family: "Noto Sans CJK TC", sans-serif; font-size:20px; }
body { margin:0; background:#fff; color:#222; } main { max-width:760px; margin:auto; padding:20px; }
pre { overflow-x:auto; background:#f3f4f6; padding:16px; font-size:13px; line-height:1.6; }
code { font-family:monospace; } table { border-collapse:collapse; width:100%; }
td,th { border:1px solid #ddd; padding:7px; } details { border:1px solid #ddd; padding:12px; }
@media(max-width:600px) { html { font-size:18px; } main { padding:16px; } }
'''
html='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><style>'+base+css+'</style><main class="md-typeset">'+body+'</main></html>'
(A/'render/section.html').write_text(html)
if '--worker' in sys.argv:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
                                  args=['--no-sandbox','--disable-dev-shm-usage'],timeout=10000)
        for width,height in [(1280,800),(390,844)]:
            page=browser.new_page(viewport=dict(width=width,height=height),device_scale_factor=1)
            page.goto((A/'render/section.html').as_uri(),wait_until='load',timeout=10000)
            page.locator('details').evaluate('(el)=>el.open=true')
            page.screenshot(path=str(A/f'render/section-{width}.png'),full_page=True,timeout=10000)
            print(json.dumps(dict(width=width,height=height,
                                 page_height=page.evaluate('document.body.scrollHeight'),
                                 body_text_length=len(page.locator('main').inner_text()),
                                 details_open=True)),flush=True)
            page.close()
        print('chromium_version='+browser.version,flush=True)
        browser.close()
else:
    command=[sys.executable,str(Path(__file__).resolve()),'--worker']
    started=time.perf_counter()
    try:
        r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
        stdout,stderr=r.stdout,r.stderr
        result=dict(exit_code=r.returncode,timed_out=False)
    except subprocess.TimeoutExpired as e:
        stdout=(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else (e.stdout or '')
        stderr=(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else (e.stderr or '')
        result=dict(exit_code=124,timed_out=True,timeout_seconds=30)
    (A/'render/chromium.stdout.txt').write_text(stdout)
    (A/'render/chromium.stderr.txt').write_text(stderr)
    result.update(command=command,elapsed_seconds=time.perf_counter()-started,
                  python=sys.version,playwright=importlib.metadata.version('playwright'),
                  markdown=importlib.metadata.version('markdown'),
                  render_scope='Only frozen 13.10; actual Chromium HTML render with course.css and local wrapper. Not a full published-site validation.')
    (A/'render/render-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print(stdout)
    print(stderr)
