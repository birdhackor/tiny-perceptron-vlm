import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT / 'docs/course-revision-20261005/continuity'
ART = BASE / 'artifacts/20_ABC'
PAGES = {p['page_id']: p for p in json.loads((BASE / 'inventory.json').read_text())['pages']}
TRACE = BASE / 'traces/20_ABC.jsonl'

def record(page_id, understanding, previous, new_question, switches=None, issues=None, unit=None, extra=None):
    p=PAGES[page_id]
    d={'recorded_at': datetime.now(timezone.utc).isoformat(), 'reviewer_task':'/root/continuity_20_abc', 'fork_turns':'none', 'page_id':page_id, 'unit':unit or p['title'], 'source_sha256':p['source_sha256'], 'figures_sha256':p['figures_sha256'], 'understanding':understanding, 'previous_teaching':previous, 'new_question':new_question, 'switch_notices':switches or [], 'issues':issues or [], 'visual_need':'文字是否足夠與圖是否必要依當節另記'}
    if extra: d.update(extra)
    TRACE.parent.mkdir(parents=True,exist_ok=True)
    with TRACE.open('a') as f: f.write(json.dumps(d,ensure_ascii=False)+'\n')
    print('RECORDED',page_id,d['recorded_at'])

def read(page_id):
    p=PAGES[page_id]
    print(json.dumps(p,ensure_ascii=False))
    print((ROOT/p['snapshot']).read_text())

def units(page_id):
    import re
    source=(ROOT/PAGES[page_id]['snapshot']).read_text()
    starts=[m.start() for m in re.finditer(r'^## ',source,re.M)]
    if not starts or starts[0]!=0: starts.insert(0,0)
    return [source[a:b] for a,b in zip(starts,starts[1:]+[len(source)])]

def read_unit(page_id,index):
    chunks=units(page_id)
    print(json.dumps(PAGES[page_id],ensure_ascii=False))
    print('UNIT',index,'OF',len(chunks),'unit_sha256',hashlib.sha256(chunks[index].encode()).hexdigest())
    print(chunks[index])

def pageview(page_id, capture=True):
    from playwright.sync_api import sync_playwright
    result=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
        for label,width,height in [('desktop',1280,900),('mobile',390,844)]:
            page=browser.new_page(viewport={'width':width,'height':height})
            url=f'http://127.0.0.1:8765/{page_id}.html'
            response=page.goto(url,wait_until='domcontentloaded',timeout=15000)
            view={'recorded_at':datetime.now(timezone.utc).isoformat(),'page_id':page_id,'url':url,'viewport':{'width':width,'height':height},'status':response.status if response else None,'screenshots':[]}
            view['rendered_cpu_outputs']=[o.inner_text() for o in page.locator('main .output').all()]
            imgs=page.locator('main img').all()
            if capture:
                for i,img in enumerate(imgs):
                    img.scroll_into_view_if_needed(timeout=15000)
                    img.evaluate("e => { e.scrollIntoView({block:'start'}); window.scrollBy(0,-65); }")
                    path=ART/f'{page_id}-{label}-{i}.png'
                    page.screenshot(path=str(path),full_page=False,timeout=15000)
                    view['screenshots'].append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'image_src':img.get_attribute('src'),'view_image_inspected':False})
                    if img.bounding_box()['height'] > height-80:
                        img.evaluate("e => { e.scrollIntoView({block:'end'}); window.scrollBy(0,15); }")
                        path=ART/f'{page_id}-{label}-{i}-bottom.png'
                        page.screenshot(path=str(path),full_page=False,timeout=15000)
                        view['screenshots'].append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'image_src':img.get_attribute('src'),'view_image_inspected':False})
            result.append(view)
            page.close()
        browser.close()
    with (ART/'pageviews.jsonl').open('a') as f:
        for v in result: f.write(json.dumps(v,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return result
