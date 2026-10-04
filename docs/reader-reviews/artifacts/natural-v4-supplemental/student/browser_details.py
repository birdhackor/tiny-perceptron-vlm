from pathlib import Path
import hashlib
import json
from playwright.sync_api import sync_playwright

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/reader-reviews/artifacts/natural-v4-supplemental/student'
BASE='http://127.0.0.1:8769/'
figure_names=['natural_shared_chat.svg','natural_photo_evidence.svg','natural_reading_order.svg','natural-v4-asr-two-routes.svg']
receipt={'figures':[], 'entry_clicks':[], 'code_blocks':[]}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    for name in figure_names:
        original=ROOT/'course/figures'/name
        raw=original.read_bytes()
        snapshot=OUT/'snapshots'/name
        snapshot.write_bytes(raw)
        url=BASE+'figures/'+name
        response=page.request.get(url)
        fetched=response.body()
        (OUT/'browser'/name).write_bytes(fetched)
        page.set_content('<html><body style="margin:0;background:white"><img style="display:block;width:900px;height:auto" src="'+url+'"></body></html>',wait_until='networkidle')
        page.locator('img').screenshot(path=str(OUT/'browser'/f'{name}.png'))
        receipt['figures'].append({'source':str(original.relative_to(ROOT)), 'source_sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw),'snapshot':str(snapshot.relative_to(ROOT)), 'url':url,'http_status':response.status,'retrieved_sha256':hashlib.sha256(fetched).hexdigest(),'source_matches_browser_retrieval':raw==fetched,'render_engine':'Chromium '+browser.version,'render_command':'.venv/bin/python docs/reader-reviews/artifacts/natural-v4-supplemental/student/browser_details.py','png':str((OUT/'browser'/f'{name}.png').relative_to(ROOT))})
    page.goto(BASE+'index.html',wait_until='networkidle')
    for label in ['第 20 章：做一位能看圖、讀字與聽問題的助理','20.2 不重新訓練，怎麼先開啟成品？']:
        link=page.locator('article').first.get_by_role('link',name=label,exact=True)
        from_url=page.url
        link.click()
        page.wait_for_load_state('networkidle')
        receipt['entry_clicks'].append({'from':from_url,'visible_label':label,'actual_destination':page.url,'destination_heading':page.locator('article h1').first.inner_text()})
    article=page.locator('article').first
    (OUT/'browser/20.2.html.txt').write_text(article.inner_text(),encoding='utf-8')
    (OUT/'browser/20.2.html.html').write_text(page.content(),encoding='utf-8')
    page.screenshot(path=str(OUT/'browser/20.2.html.png'),full_page=True)
    guide=article.locator('a').filter(has_text='學生操作指引').first
    if guide.count()==0:
        guide=article.locator('a[href$="natural-v4-student.html"]').first
    label=guide.inner_text()
    from_url=page.url
    guide.click()
    page.wait_for_load_state('networkidle')
    page.screenshot(path=str(OUT/'browser/20.2-student-link-external-destination.png'),full_page=True)
    (OUT/'browser/20.2-student-link-external-destination.html').write_text(page.content(),encoding='utf-8')
    receipt['entry_clicks'].append({'from':from_url,'visible_label':label,'actual_destination':page.url,'destination_title':page.title(),'destination_headings':page.locator('h1').all_inner_texts(),'destination_article_count':page.locator('article').count(),'scope_note':'Only destination identity inspected; external source is not assumed to be this current canonical guide.'})
    page.goto(BASE+'natural-v4-student.html',wait_until='networkidle')
    article=page.locator('article').first
    for i, pre in enumerate(article.locator('pre').all()):
        pre.scroll_into_view_if_needed()
        pre.evaluate('(e)=>e.scrollLeft=e.scrollWidth')
        (OUT/'browser'/f'code-block-{i:02d}.txt').write_text(pre.inner_text(),encoding='utf-8')
        pre.screenshot(path=str(OUT/'browser'/f'code-block-{i:02d}-right.png'))
        receipt['code_blocks'].append({'index':i, 'text':pre.inner_text(), 'dimensions':pre.evaluate('(e)=>({clientWidth:e.clientWidth,scrollWidth:e.scrollWidth,scrollLeft:e.scrollLeft,height:e.clientHeight})')})
    (OUT/'browser/details-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    browser.close()
