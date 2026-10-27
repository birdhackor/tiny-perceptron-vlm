"""Render frozen SVG and current local canonical page; PNGs stay outside Git."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=Path('/workspace/work/tokenizer-dynamic-20261027/readme-callback')
EVIDENCE=ROOT/'docs/technical-reviews/artifacts/tokenizer-20261027/readme-callback/render-readme-01.json'
svg=ROOT/'course/figures/rewrite-01-character-ids.svg'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':680,'height':760},device_scale_factor=1)
    page.set_content('<html><body style="margin:0">'+svg.read_text()+'</body></html>')
    page.evaluate('document.fonts.ready')
    target=OUT/'figure-01.png'
    page.locator('svg').screenshot(path=str(target))
    rows.append({'kind':'figure','source':str(svg.relative_to(ROOT)),'source_sha256':sha(svg),'png':str(target),'png_sha256':sha(target),'dimensions':{'width':640,'height':710},'method':'Playwright page.set_content of confirmed SVG bytes, screenshot SVG element; inspect separately with view_image.'})
    page.close()
    for width,height,label in [(1440,1100,'desktop'),(390,844,'mobile')]:
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1,reduced_motion='reduce')
        response=page.goto('http://127.0.0.1:8000/readme.html',wait_until='networkidle')
        page.evaluate('document.fonts.ready')
        main=page.locator('.md-content')
        text=main.inner_text()
        target=OUT/f'page-{label}-01.png'
        main.screenshot(path=str(target))
        rows.append({'kind':'page','url':page.url,'http_status':response.status,'viewport':{'width':width,'height':height},'png':str(target),'png_sha256':sha(target),'body_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'main_bounds':main.bounding_box(),'has_dryrun': 'dry-run-no-weight-update' in text,'has_313': '313' in text,'has_historical_limit': 'Colab 雲端執行尚未實機驗證' in text,'figure_bounds':page.locator('.md-content img').first.bounding_box(),'horizontal_overflow':page.evaluate('document.documentElement.scrollWidth > innerWidth'),'method':'Capture current final local site canonical readme.html main content with reduced motion; inspect PNG separately.'})
        if label=='mobile':
            image=page.locator('.md-content img').first
            image.scroll_into_view_if_needed()
            target=OUT/'page-mobile-figure-01.png'
            page.screenshot(path=str(target))
            rows.append({'kind':'mobile_figure_context','png':str(target),'png_sha256':sha(target),'viewport':{'width':width,'height':height}})
        page.close()
    browser.close()
receipt={'schema_version':1,'owner':'/root/tech_readme_callback_20261027','created_at':datetime.now(timezone.utc).isoformat(),'readme_sha256':sha(ROOT/'README.md'),'reading_times_sha256':sha(ROOT/'course/reading-times.json'),'site_html_sha256':sha(ROOT/'outputs/tokenizer-20261027/site/readme.html'),'renders':rows,'visual_inspection':'Pending explicit view_image inspection by this reviewer; conclusions are recorded in the technical report.'}
EVIDENCE.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
