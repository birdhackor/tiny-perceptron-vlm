"""Own Inkscape and real-page Chromium render; only current9.10, no author summaries."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
records=[]
for name in ['rewrite-09-10-calibration-bins.svg','audio_calibration_test.svg']:
    for width in [640,360]:
        destination=BASE/'renders'/f'{Path(name).stem}-{width}.png'
        cmd=['inkscape',str(ROOT/'course/figures'/name),'--export-type=png',f'--export-width={width}',f'--export-filename={destination}']
        completed=subprocess.run(cmd,capture_output=True,text=True,timeout=30,check=False)
        records.append({'command_argv':cmd,'exit_code':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,'artifact':destination.relative_to(ROOT).as_posix()})
        assert completed.returncode==0 and destination.exists()

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--disable-gpu','--renderer-process-limit=1'])
    for width,height in [(1280,800),(390,844)]:
        context=browser.new_context(viewport={'width':width,'height':height},device_scale_factor=1)
        context.route('**/*',lambda route: route.continue_() if urlsplit(route.request.url).hostname in ['127.0.0.1','localhost'] else route.abort())
        page=context.new_page();responses=[]
        page.on('response',lambda response: responses.append(response))
        response=page.goto('http://127.0.0.1:8765/9.10.html',wait_until='networkidle',timeout=30000)
        assert response and response.status==200
        page.locator('details').evaluate_all('(nodes)=>nodes.forEach(n=>n.open=true)')
        page.evaluate('document.fonts.ready');page.wait_for_timeout(200)
        images=page.locator('img'); image_facts=images.evaluate_all('(nodes)=>nodes.map(n=>({src:n.getAttribute("src"),complete:n.complete,naturalWidth:n.naturalWidth,naturalHeight:n.naturalHeight,width:n.getBoundingClientRect().width,height:n.getBoundingClientRect().height}))')
        assert all(x['complete'] and x['naturalWidth']>0 for x in image_facts)
        for name in ['rewrite-09-10-calibration-bins.svg','audio_calibration_test.svg']:
            locator=page.locator(f'img[src$="{name}"]');assert locator.count()==1
            locator.screenshot(path=str(BASE/'renders'/f'page-{width}-{Path(name).stem}.png'))
        audio=page.locator('img[src$="audio_calibration_test.svg"]')
        audio.evaluate('(n)=>window.scrollTo(0,n.getBoundingClientRect().top+window.scrollY-90)')
        page.wait_for_timeout(100)
        page.screenshot(path=str(BASE/'renders'/f'page-{width}-audio-top-viewport.png'))
        if width==1280:
            audio.evaluate('(n)=>window.scrollTo(0,n.getBoundingClientRect().bottom+window.scrollY-window.innerHeight+20)')
            page.wait_for_timeout(100)
            page.screenshot(path=str(BASE/'renders'/f'page-{width}-audio-bottom-viewport.png'))
        page.evaluate('window.scrollTo(0,0)')
        page.wait_for_timeout(100)
        page.screenshot(path=str(BASE/'renders'/f'page-{width}-full.png'),full_page=True)
        (BASE/'inputs'/f'page-{width}-current.html').write_text(page.content())
        resource_facts=[]
        for res in responses:
            if any(res.url.endswith(n) for n in ['rewrite-09-10-calibration-bins.svg','audio_calibration_test.svg']):
                body=res.body();name=res.url.rsplit('/',1)[1];digest=hashlib.sha256(body).hexdigest()
                assert digest==hashlib.sha256((ROOT/'course/figures'/name).read_bytes()).hexdigest()
                resource_facts.append({'url':res.url,'status':res.status,'sha256':digest,'bytes':len(body)})
        assert len(resource_facts)==2
        records.append({'action':'Actual page.goto and screenshot','url':page.url,'http_status':response.status,'viewport':{'width':width,'height':height},'details_opened':True,'images':image_facts,'served_figure_hashes':resource_facts,'body_scroll_width':page.evaluate('document.body.scrollWidth'),'viewport_width':width,'scroll_context':'Additional normal-scroll top/bottom viewport screenshots show graph labels outside sticky navbar; full-page screenshot resets scroll0. No CSS or content modified; only details opened and scroll position changed.','source':'Own browser rendering and served SVG hashes, not root parity receipt.'})
        context.close()
    browser_version=browser.version;browser.close()
receipt={'environment':{'python':platform.python_version(),'inkscape':subprocess.check_output(['inkscape','--version'],text=True).strip(),'chromium':browser_version,'device':'CPU/headless; --disable-gpu'},'records':records}
(BASE/'execution/render-and-page-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
