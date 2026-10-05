import sys,json,pathlib,datetime,hashlib
from playwright.sync_api import sync_playwright
BASE=pathlib.Path('/workspace/tiny-perceptron-vlm/docs/course-revision-20261005/continuity/artifacts/06_09')
page_id=sys.argv[1]
suffix='-'+sys.argv[2] if len(sys.argv)>2 else ''
rows=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
    for name,width,height in [('desktop',1280,900),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height})
        url='http://127.0.0.1:8765/'+page_id+'.html'
        response=page.goto(url,wait_until='domcontentloaded',timeout=15000)
        page.locator('main details').evaluate_all('(els)=>els.forEach(e=>{if(e.querySelector("img"))e.open=true})')
        page.locator('main img').first.wait_for(state='attached',timeout=15000)
        imgs=page.locator('main img')
        for i in range(imgs.count()):
            img=imgs.nth(i);img.scroll_into_view_if_needed(timeout=15000)
            box=img.bounding_box();top=box['y']+page.evaluate('window.scrollY')
            offsets=[('',0)]
            if box['height']>height-70: offsets.append(('-bottom',box['height']-(height-80)))
            if suffix=='-bottomonly': offsets=offsets[1:]
            for portion,offset in offsets:
                page.evaluate('(y)=>window.scrollTo(0,Math.max(0,y-60))',top+offset)
                path=BASE/'screens'/f'{page_id}-{name}-{i}{suffix}{portion}.png'
                page.screenshot(path=str(path),full_page=False,timeout=15000)
                row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'page_id':page_id,'url':url,'http_status':response.status,'viewport':{'width':width,'height':height},'figure_index':i,'portion':portion or 'top','figure_src':img.get_attribute('src'),'screenshot_path':str(path),'screenshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'image_box':img.bounding_box(),'viewed':False}
                rows.append(row)
        page.close()
    browser.close()
with (BASE/'pageviews.jsonl').open('a') as f:
    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
print(json.dumps(rows,ensure_ascii=False,indent=2))
