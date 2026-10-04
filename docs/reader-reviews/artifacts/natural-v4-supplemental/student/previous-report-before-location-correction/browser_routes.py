from pathlib import Path
import hashlib
import json
from playwright.sync_api import sync_playwright

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/reader-reviews/artifacts/natural-v4-supplemental/student'
BASE = 'http://127.0.0.1:8769/'
routes = ['index.html', 'course.html', 'chapter-20.html', '20.1.html', '20.3.html', '20.9.html', '20.10.html', '20.11.html', '20.12.html', '20.13.html', 'natural-v4-data.html', 'natural-v4-training.html']
receipts = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width':1440, 'height':1000}, device_scale_factor=1)
    for name in routes:
        response = page.goto(BASE + name, wait_until='networkidle')
        article = page.locator('article').first
        text = article.inner_text()
        (OUT / f'browser/{name}.txt').write_text(text, encoding='utf-8')
        (OUT / f'browser/{name}.html').write_text(page.content(), encoding='utf-8')
        page.screenshot(path=str(OUT / f'browser/{name}.png'), full_page=True)
        receipt = {'requested_url':BASE + name, 'final_url':page.url, 'http_status':response.status,
                   'title':page.title(), 'headings':article.locator('h1,h2,h3').all_inner_texts(),
                   'links':article.locator('a').evaluate_all('(es)=>es.map(e=>({label:e.innerText,href:e.href}))'),
                   'images':article.locator('img').evaluate_all('(es)=>es.map(e=>({alt:e.alt,src:e.src,complete:e.complete,width:e.naturalWidth,height:e.naturalHeight}))'),
                   'rendered_text_sha256':hashlib.sha256(text.encode()).hexdigest()}
        for i, img in enumerate(article.locator('img').all()):
            img.screenshot(path=str(OUT / f'browser/{name}-figure-{i}.png'))
        receipts.append(receipt)
    page.goto(BASE + 'natural-v4-student.html', wait_until='networkidle')
    article = page.locator('article').first
    (OUT / 'browser/student-article.txt').write_text(article.inner_text(), encoding='utf-8')
    click_records = []
    for label in ['20.1 的輸入分工','20.13','20.3','20.9','20.10','20.11','20.12','資料說明','訓練指引']:
        page.goto(BASE + 'natural-v4-student.html', wait_until='networkidle')
        link = page.locator('article').first.get_by_role('link', name=label, exact=True)
        original_href = link.get_attribute('href')
        link.click()
        page.wait_for_load_state('networkidle')
        click_records.append({'visible_label':label,'href':original_href,'actual_destination':page.url,
                              'destination_heading':page.locator('article h1').first.inner_text()})
    page.goto(BASE + 'index.html', wait_until='networkidle')
    entry_links = page.locator('article').first.locator('a').evaluate_all('(es)=>es.map(e=>({label:e.innerText,href:e.href}))')
    print(json.dumps({'entry_links':entry_links,'pages':[{'url':r['final_url'],'status':r['http_status'],'headings':r['headings'],'images':r['images']} for r in receipts], 'clicked':click_records},ensure_ascii=False,indent=2))
    (OUT / 'browser/routes-receipt.json').write_text(json.dumps({'browser':browser.version,'engine':'Chromium','executable':'/usr/bin/chromium','viewport':{'width':1440,'height':1000},'pages':receipts,'actual_clicks':click_records},indent=2,ensure_ascii=False)+'\n')
    browser.close()
