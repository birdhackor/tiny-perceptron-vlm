from pathlib import Path
from playwright.sync_api import sync_playwright
import json, hashlib, datetime, sys
root=Path('/workspace/tiny-perceptron-vlm')
out=root/'docs/course-revision-20261005/continuity/artifacts/17_19'
pid=sys.argv[1]
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage','--disable-background-networking'])
    for label,w,h in [('desktop',1280,900),('mobile',390,844)]:
        page=b.new_page(viewport={'width':w,'height':h})
        url=f'http://127.0.0.1:8765/{pid}.html'
        page.goto(url,wait_until='domcontentloaded',timeout=15000)
        figures=page.locator('article img, .md-content img')
        count=figures.count()
        if not count: figures=page.locator('img');count=figures.count()
        for n in range(count):
            item=figures.nth(n)
            item.scroll_into_view_if_needed(timeout=15000)
            path=out/f'{pid}-{label}-{n}.png'
            page.screenshot(path=str(path),full_page=False,timeout=15000)
            data={'page_id':pid,'url':url,'viewport':{'width':w,'height':h},'figure_index':n,'image_src':item.get_attribute('src'),'screenshot':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            with (out/'pageviews.jsonl').open('a') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')
            print(json.dumps(data,ensure_ascii=False))
        page.close()
    b.close()
