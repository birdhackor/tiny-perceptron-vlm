import json, hashlib, math, difflib
from pathlib import Path
from urllib.parse import unquote
from playwright.sync_api import sync_playwright

p = Path('docs/reader-reviews/artifacts/natural-v4-supplemental/training-course')
q = p / 'round5'
base = 'http://127.0.0.1:8789/'
H = lambda b: hashlib.sha256(b).hexdigest()
r = {'reviewer_task': '/root/v4_review_coordinator/reader_whole_training_course', 'round': 5,
     'real_browser': 'existing .venv Playwright Chromium', 'executable_path': '/usr/bin/chromium',
     'base': base, 'reported_literal_revision': '160fd47dede5c2453c8a08dd535290ddfd0b915d',
     'training': {}, 'prerequisites': [], 'actual_links': [], 'toc_clicks': [], 'guides': [], 'navigation': []}
def save():
    (q/'browser-receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
def capture(response, name):
    raw = response.body()
    fn = q/(name+'.http-response.html'); fn.write_bytes(raw)
    matches = [str(x) for x in Path('outputs').rglob(name+'.html') if x.read_bytes()==raw]
    assert matches, name+' actual HTTP bytes failed to match any local built page'
    return {'actual_URL': response.url, 'status': response.status, 'HTTP_raw_sha256': H(raw),
            'HTTP_bytes': len(raw), 'HTTP_snapshot': str(fn),
            'exact_local_build_byte_matches': matches}
def tiles(name, start, end=None):
    article = page.locator('article'); box = article.bounding_box()
    sy = page.evaluate('window.scrollY'); sb=start.bounding_box()
    top=sb['y']+sy
    bottom=end.bounding_box()['y']+sy if end else box['y']+sy+box['height']
    out=[]
    for n, offset in enumerate(range(0,math.ceil(bottom-top),2400),1):
        page.evaluate('(y)=>window.scrollTo(0,y)',top+offset-96); page.wait_for_timeout(60)
        fn=q/(name+f'-tile-{n:02}.png')
        page.screenshot(path=str(fn),clip={'x':box['x'],'y':96,'width':box['width'],'height':2600})
        out.append({'order':n,'offset':offset,'screenshot':str(fn)})
    return out

with sync_playwright() as w:
    browser=w.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':2700},device_scale_factor=1)
    response=page.goto(base+'16.1.html',wait_until='networkidle')
    text=page.locator('article').inner_text(); (q/'16.1.rendered-visible-article.txt').write_text(text)
    r['prerequisites'].append({'section':'16.1','HTTP':capture(response,'16.1'),
        'actual_URL':page.url,'h1':page.locator('article h1').all_inner_texts(),
        'full_visible_text_sha256':H(text.encode()),'full_visible_text':str(q/'16.1.rendered-visible-article.txt'),
        'CPU_output_widgets':page.locator('article [class*="output"]').evaluate_all('(els)=>els.map(e=>({class:e.className,text:e.innerText}))'),
        'screens':tiles('16.1',page.locator('article h1').first)})
    link=page.locator('article a').filter(has_text='T.8的效率量測方法').first
    link.scroll_into_view_if_needed()
    origin={'from':page.url,'visible_label':link.inner_text(),'href':link.get_attribute('href')}
    assert unquote(origin['href']).endswith('training.html#選讀效率實驗的量測條件')
    with page.expect_navigation(wait_until='networkidle') as result: link.click()
    response=result.value; article=page.locator('article')
    current=article.inner_text(); old=(p/'round4/training.rendered-visible-article.txt').read_text()
    (q/'training.rendered-visible-article.txt').write_text(current);(q/'training.rendered-dom.html').write_text(page.content())
    diff=''.join(difflib.unified_diff(old.splitlines(keepends=True),current.splitlines(keepends=True),fromfile='own-round4-complete-visible',tofile='round5-current-visible',n=2))
    (q/'training-visible-change.diff').write_text(diff)
    heading=article.locator('[id="選讀效率實驗的量測條件"]'); assert heading.count()==1
    page.set_viewport_size({'width':1440,'height':1100});heading.scroll_into_view_if_needed();page.wait_for_timeout(80)
    fn=q/'16.1-to-measurement-heading.png';page.screenshot(path=str(fn))
    origin.update({'actual_destination':page.url,'unquoted_destination':unquote(page.url),
        'actual_target_heading':heading.inner_text(),'actual_target_tag':heading.evaluate('(e)=>e.tagName'),
        'parent_T8_heading':article.locator('[id="T.8"]').inner_text(),'screenshot':str(fn)})
    r['actual_links'].append(origin)
    assert origin['actual_target_tag']=='H3'
    r['training']={'HTTP':capture(response,'training'),'actual_URL':page.url,
        'h1':article.locator('h1').all_inner_texts(),'full_visible_text_sha256':H(current.encode()),
        'old_own_full_visible_sha256':H(old.encode()),'full_visible_text_delta':str(q/'training-visible-change.diff'),
        'paragraphs':article.locator('p').count(),'tables':article.locator('table').count(),'code_blocks':article.locator('pre').count(),
        'SVG_count':article.locator('img[src$=".svg"]').count(),'output_widgets':article.locator('[class*="output"]').count(),
        'actual_revision_pinned_hrefs':article.locator('a').evaluate_all('(els)=>els.map(e=>e.getAttribute("href")).filter(h=>h&&h.includes("160fd47dede5c2453c8a08dd535290ddfd0b915d"))')}
    t8=current.split('架構比較一次只換一個條件¶',1)[1].split('真的縮小保存格式，再量品質¶',1)[0]
    (q/'T8.rendered-visible-complete.txt').write_text('架構比較一次只換一個條件¶'+t8)
    page.set_viewport_size({'width':1440,'height':2700})
    r['training']['T8_screens']=tiles('T8',article.locator('[id="T.8"]'),article.locator('[id="T.9"]'))
    page.set_viewport_size({'width':1440,'height':1100})
    for i in range(1,12):
        link=page.locator(f'a[href$="#T.{i}"]').filter(visible=True).first
        label=link.inner_text();href=link.get_attribute('href');link.click();page.wait_for_timeout(50)
        r['toc_clicks'].append({'visible_label':label,'href':href,'actual_destination':page.url,
            'target_heading':article.locator(f'[id="T.{i}"]').inner_text(),'target_id':f'T.{i}'})
    save()
    page.locator('a[href$="#T.8"]').filter(visible=True).first.click();page.wait_for_timeout(60)
    link=page.locator('nav a').filter(has_text='選讀：效率實驗的量測條件').filter(visible=True).first
    label=link.inner_text();href=link.get_attribute('href');link.click();page.wait_for_timeout(60)
    fn=q/'measurement-subheading-TOC.png';page.screenshot(path=str(fn))
    r['actual_links'].append({'from':'current training right TOC','visible_label':label,'href':href,
        'actual_destination':page.url,'target_heading':heading.inner_text(),'screenshot':str(fn)})
    link=article.locator('a').filter(has_text='本節入口').first;link.scroll_into_view_if_needed()
    label=link.inner_text();href=link.get_attribute('href');link.click();page.wait_for_timeout(60)
    r['actual_links'].append({'from':'measurement conditions paragraph','visible_label':label,'href':href,
        'actual_destination':page.url,'target_heading':article.locator('[id="T.8"]').inner_text()})
    link=article.locator('a').filter(has_text='16.1').first;link.scroll_into_view_if_needed()
    label=link.inner_text();href=link.get_attribute('href')
    with page.expect_navigation(wait_until='networkidle') as result: link.click()
    r['actual_links'].append({'from':'T8 source explicit first bottleneck prerequisite','visible_label':label,'href':href,
        'actual_destination':page.url,'target_heading':page.locator('article h1').inner_text()})
    link=page.locator('article a').filter(has_text='矩陣使用量不等於耗時').first
    label=link.inner_text();href=link.get_attribute('href');link.scroll_into_view_if_needed()
    with page.expect_navigation(wait_until='networkidle') as result: link.click()
    response=result.value;text=page.locator('article').inner_text()
    (q/'15.11.rendered-visible-article.txt').write_text(text)
    r['actual_links'].append({'from':'16.1 needed timing prerequisite','visible_label':label,'href':href,
        'actual_destination':page.url,'target_heading':page.locator('article h1').inner_text()})
    page.set_viewport_size({'width':1440,'height':2700})
    r['prerequisites'].append({'section':'15.11','HTTP':capture(response,'15.11'),'actual_URL':page.url,
        'h1':page.locator('article h1').all_inner_texts(),'full_visible_text_sha256':H(text.encode()),
        'full_visible_text':str(q/'15.11.rendered-visible-article.txt'),
        'CPU_output_widgets':page.locator('article [class*="output"]').evaluate_all('(els)=>els.map(e=>({class:e.className,text:e.innerText}))'),
        'screens':tiles('15.11',page.locator('article h1').first)})
    save()
    page.set_viewport_size({'width':1440,'height':1100})
    prior=json.loads((p/'round4/browser-receipt.json').read_text())
    for name,label in [('natural-v4-student','操作指引'),('natural-v4-data','資料與來源'),('natural-v4-training','自行重訓')]:
        page.goto(base+'training.html',wait_until='networkidle')
        page.get_by_role('navigation',name='標籤頁').get_by_role('link',name='教材',exact=True).click();page.wait_for_load_state('networkidle')
        page.locator('label[for="__nav_3_20"]').filter(has_text='第 20 章').first.click()
        link=page.locator('nav a[href$="'+name+'.html"]').filter(visible=True).first
        link.scroll_into_view_if_needed();actual=link.inner_text();href=link.get_attribute('href')
        with page.expect_navigation(wait_until='networkidle') as result: link.click()
        text=page.locator('article').inner_text();opening=text.split('1.')[0]
        before=next(x for x in prior['guides'] if x['visible_label']==label)
        (q/(name+'.landing-visible-opening.txt')).write_text(opening)
        r['guides'].append({'visible_label':actual,'href':href,'actual_destination':page.url,
            'h1':page.locator('article h1').all_inner_texts(),'opening':opening,'opening_exact_own_round4':opening==before['opening'],
            'HTTP':capture(result.value,name),'scope':'actual landing/opening/navigation only; previous personally viewed unchanged landing image reused, no full guide read claim'})
    for css,name in [('.md-footer__link--prev','glossary'),('.md-footer__link--next','chapter-01')]:
        page.goto(base+'training.html',wait_until='networkidle');link=page.locator(css)
        link.scroll_into_view_if_needed();label=link.inner_text();href=link.get_attribute('href')
        with page.expect_navigation(wait_until='networkidle') as result: link.click()
        r['navigation'].append({'visible_label':label,'href':href,'actual_destination':page.url,
            'h1':page.locator('article h1').all_inner_texts(),'HTTP':capture(result.value,name)})
    save();browser.close()
print(json.dumps({'HTTP':r['training']['HTTP'],'actual_links':r['actual_links'],
    'TOC_count':len(r['toc_clicks']),'prereq_CPU':[{k:x[k] for k in ('section','CPU_output_widgets')} for x in r['prerequisites']],
    'guides':[{'label':x['visible_label'],'heading':x['h1'],'opening':x['opening'],'same_opening':x['opening_exact_own_round4']} for x in r['guides']]},ensure_ascii=False,indent=2))
