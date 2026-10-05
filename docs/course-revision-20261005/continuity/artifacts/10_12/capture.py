import datetime, hashlib, json, pathlib, sys
from playwright.sync_api import sync_playwright
page_id=sys.argv[1]
base=pathlib.Path('/workspace/tiny-perceptron-vlm/docs/course-revision-20261005/continuity/artifacts/10_12')
records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
    for label,width,height in [('desktop',1280,900),('mobile',390,844)]:
        context=browser.new_context(viewport={'width':width,'height':height},device_scale_factor=1)
        page=context.new_page()
        url=f'http://127.0.0.1:8765/{page_id}.html'
        page.goto(url,wait_until='domcontentloaded',timeout=15000)
        page.locator('details').evaluate_all('(nodes) => nodes.forEach(n => n.open = true)')
        images=page.locator('main img, article img')
        count=images.count()
        if not count:
            images=page.locator('img');count=images.count()
        targets=list(range(count)) if count else [None]
        for index in targets:
            box=None
            if index is not None:
                img=images.nth(index)
                img.scroll_into_view_if_needed(timeout=15000)
                source=img.get_attribute('src')
                box=img.bounding_box()
            else:
                source=None
            path=base/f'{page_id}-{label}-{index if index is not None else "page"}.png'
            page.screenshot(path=str(path),full_page=False,timeout=15000)
            records.append({'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'page_id':page_id,'url':url,'viewport':{'width':width,'height':height},'image_index':index,'image_source':source,'screenshot':str(path),'screenshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'viewed':False})
            if box and box['height'] > height-130:
                for suffix, align in [('top',0),('bottom',1)]:
                    img.evaluate('(n,a) => window.scrollTo(0,window.scrollY+n.getBoundingClientRect().top-(a ? innerHeight-n.getBoundingClientRect().height-15 : 85))',align)
                    path=base/f'{page_id}-{label}-{index}-{suffix}.png'
                    page.screenshot(path=str(path),full_page=False,timeout=15000)
                    records.append({'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'page_id':page_id,'url':url,'viewport':{'width':width,'height':height},'image_index':index,'image_source':source,'screenshot':str(path),'screenshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'viewed':False})
        context.close()
    browser.close()
with (base/'pageviews.jsonl').open('a') as f:
    for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps(records,ensure_ascii=False))
