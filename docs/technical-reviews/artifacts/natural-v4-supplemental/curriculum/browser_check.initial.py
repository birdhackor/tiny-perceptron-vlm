from pathlib import Path
import hashlib, json, urllib.request
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8783/'
receipts=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1280,'height':960},device_scale_factor=1)
    response=page.goto(BASE+'curriculum.html',wait_until='networkidle')
    page.locator('article h1').wait_for()
    page.screenshot(path=str(HERE/'browser.curriculum-top.png'))
    receipts.append({'action':'navigate curriculum','url':page.url,'status':response.status,'h1':page.locator('article h1').inner_text(),'visible_intro':page.locator('article').inner_text()[:1400]})
    # Save source provenance only, distinct from personal visual inspection.
    html=urllib.request.urlopen(BASE+'curriculum.html').read()
    (HERE/'browser.curriculum.html').write_bytes(html)
    receipts.append({'action':'HTTP provenance','html_sha256':hashlib.sha256(html).hexdigest(),'source_revision_visible':'f8ca78bc3ffef82b0faa352c464909b76bfd8976' in html.decode(),'current_document_sha256':hashlib.sha256((ROOT/'docs/curriculum.md').read_bytes()).hexdigest()})
    page.locator('article a',has_text='20.2').first.click()
    page.wait_for_load_state('networkidle')
    receipts.append({'action':'click curriculum 20.2','url':page.url,'h1':page.locator('article h1').inner_text(),'visible_output':page.locator('article').inner_text()[:1200]})
    page.goto(BASE+'curriculum.html',wait_until='networkidle')
    page.locator('article a',has_text='教材入口').first.click()
    page.wait_for_load_state('networkidle')
    page.locator('article a',has_text='第一節').first.click()
    page.wait_for_load_state('networkidle')
    receipts.append({'action':'click curriculum entry then first lesson','url':page.url,'h1':page.locator('article h1').inner_text(),'visible_output':page.locator('article').inner_text()[:1800],'notebook_links':page.locator('article a').evaluate_all("els=>els.filter(e=>e.href.includes('ipynb')||e.href.includes('colab')).map(e=>({text:e.innerText,url:e.href}))")})
    page.screenshot(path=str(HERE/'browser.first-lesson.png'))
    page.goto(BASE+'figures/curriculum_learning_flow.svg',wait_until='networkidle')
    page.screenshot(path=str(HERE/'curriculum_learning_flow.render.png'),full_page=True)
    raw=urllib.request.urlopen(BASE+'figures/curriculum_learning_flow.svg').read()
    local=(ROOT/'course/figures/curriculum_learning_flow.svg').read_bytes()
    receipts.append({'action':'Chromium SVG render','url':page.url,'server_sha256':hashlib.sha256(raw).hexdigest(),'local_sha256':hashlib.sha256(local).hexdigest(),'byte_equal':raw==local,'text_labels':page.locator('svg text').all_text_contents()})
    receipts.append({'browser':browser.version,'viewport':'1280x960','renderer':'Playwright Chromium, headless'})
    browser.close()
(HERE/'browser.receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipts,ensure_ascii=False,indent=2))
