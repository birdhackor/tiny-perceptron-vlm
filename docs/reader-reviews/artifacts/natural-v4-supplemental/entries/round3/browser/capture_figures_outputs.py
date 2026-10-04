from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
root=Path('docs/reader-reviews/artifacts/natural-v4-supplemental/entries/round3');b=root/'browser';base='http://127.0.0.1:8784/'
figures=[];outputs=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium');page=browser.new_page(viewport={'width':1440,'height':1000});page.set_default_timeout(25000)
    for slug,name in [('readme','character_ids'),('20.1','natural_shared_chat')]:
        r=page.goto(base+slug+'.html',wait_until='networkidle')
        img=page.locator('img[src$="'+name+'.svg"]:visible').first;img.scroll_into_view_if_needed();img.screenshot(path=str(b/f'{name}-render.png'))
        url=img.evaluate('(e)=>e.src');rsvg=page.request.get(url);raw=rsvg.body();(b/f'{name}-served.svg').write_bytes(raw)
        source=Path('course/figures')/(name+'.svg');snapshot=root/'read-snapshots'/source;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(source.read_bytes())
        previous=Path('docs/reader-reviews/artifacts/natural-v4-supplemental/entries/round2/read-snapshots')/source
        figures.append({'file':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_bytes':source.stat().st_size,'snapshot':str(snapshot),'actual_page':page.url,'served_url':url,'served_status':rsvg.status,'served_sha256':hashlib.sha256(raw).hexdigest(),'served_bytes':len(raw),'exact_source_served_equal':source.read_bytes()==raw,'exact_round2_source_equal':source.read_bytes()==previous.read_bytes(),'render':str(b/f'{name}-render.png'),'render_method':'New current 8784 Playwright Chromium render of visible in-page img; scroll_into_view_if_needed and element screenshot','browser_version':browser.version,'newly_rendered':True})
    for slug in ['1.1','20.1']:
        r=page.goto(base+slug+'.html',wait_until='networkidle');raw=r.body();(b/f'{slug}-output-page-http.html').write_bytes(raw)
        content=page.locator('.md-content').inner_text();(b/f'{slug}-output-page-content.txt').write_text(content)
        label=page.get_by_text('實際執行結果 · CPU',exact=True).first;block=label.locator('..');block.scroll_into_view_if_needed();block.screenshot(path=str(b/f'{slug}-actual-CPU-output.png'))
        outputs.append({'lesson':slug,'actual_url':page.url,'title':page.title(),'label':label.inner_text(),'actual_visible_block':block.inner_text(),'screenshot':str(b/f'{slug}-actual-CPU-output.png'),'raw_evidence':str(b/f'{slug}-output-page-http.html'),'http_status':r.status,'http_sha256':hashlib.sha256(raw).hexdigest(),'http_bytes':len(raw),'content_evidence':str(b/f'{slug}-output-page-content.txt'),'provenance':'Personally navigated Chromium page; actual rendered output widget screenshot and DOM text; the displayed export execution is not a reader rerun'})
    browser.close()
(b/'figure-render-receipts.json').write_text(json.dumps(figures,ensure_ascii=False,indent=2));(b/'CPU-output-receipts.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2))
print(json.dumps({'figures':figures,'outputs':outputs},ensure_ascii=False,indent=2))
