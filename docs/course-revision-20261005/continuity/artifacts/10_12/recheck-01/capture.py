import datetime, hashlib, json, pathlib, sys
from playwright.sync_api import sync_playwright
page_id=sys.argv[1]
base=pathlib.Path('/workspace/tiny-perceptron-vlm/docs/course-revision-20261005/continuity/artifacts/10_12/recheck-01')
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
        for index in range(images.count()):
            img=images.nth(index)
            img.scroll_into_view_if_needed(timeout=15000)
            source=img.get_attribute('src')
            box=img.bounding_box()
            poses=['main','top','bottom'] if box and box['height']>height-130 else ['main']
            for pose in poses:
                if pose!='main':
                    img.evaluate('(n,a) => window.scrollTo(0,window.scrollY+n.getBoundingClientRect().top-(a ? innerHeight-n.getBoundingClientRect().height-15 : 85))',int(pose=='bottom'))
                path=base/f'{page_id}-{label}-{index}-{pose}.png'
                page.screenshot(path=str(path),full_page=False,timeout=15000)
                records.append({'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'page_id':page_id,'url':url,'viewport':{'width':width,'height':height},'image_index':index,'image_source':source,'pose':pose,'screenshot':str(path),'screenshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'viewed':False})
        context.close()
    browser.close()
with (base/'pageviews.jsonl').open('a') as f:
    for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps({'page_id':page_id,'files':[pathlib.Path(r['screenshot']).name for r in records]},ensure_ascii=False))
