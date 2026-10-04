from pathlib import Path
import json, hashlib
from playwright.sync_api import sync_playwright
root=Path('docs/reader-reviews/artifacts/natural-v4-supplemental/entries/round3/browser')
base='http://127.0.0.1:8784/'
receipts=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium')
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.set_default_timeout(25000)
    for slug in ['index','readme','course','training-assets']:
        r=page.goto(base+slug+'.html',wait_until='networkidle')
        raw=r.body();(root/f'{slug}-http.html').write_bytes(raw)
        content=page.locator('.md-content').inner_text();(root/f'{slug}-content.txt').write_text(content)
        body=page.locator('body').inner_text();(root/f'{slug}-body.txt').write_text(body)
        page.screenshot(path=str(root/f'{slug}-viewport.png'))
        page.screenshot(path=str(root/f'{slug}-full.png'),full_page=True)
        receipt={'page':slug,'url':page.url,'status':r.status,'title':page.title(),'headings':page.locator('.md-content :is(h1,h2,h3)').all_text_contents(),'links':page.locator('.md-content a').evaluate_all('(els)=>els.map(e=>({label:e.innerText.trim(),href:e.href}))'),'visible_sidebar_links':page.locator('nav a:visible').evaluate_all('(els)=>els.map(e=>({label:e.innerText.trim(),href:e.href}))'),'images':page.locator('.md-content img').evaluate_all('(els)=>els.map(e=>({alt:e.alt,src:e.src}))'),'http_bytes':len(raw),'http_sha256':hashlib.sha256(raw).hexdigest(),'content_evidence':str(root/f'{slug}-content.txt'),'screenshot':str(root/f'{slug}-full.png'),'raw_evidence':str(root/f'{slug}-http.html'),'provenance':'Playwright Chromium page.goto networkidle; navigation response.body; DOM inner_text; actual browser screenshots','browser_version':browser.version}
        receipts.append(receipt)
    browser.close()
(root/'page-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2))
print(json.dumps([{'page':r['page'],'title':r['title'],'headings':r['headings'],'links':r['links'],'images':r['images'],'status':r['status'],'http_sha256':r['http_sha256']} for r in receipts],ensure_ascii=False,indent=2))
