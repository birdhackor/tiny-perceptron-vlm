"""Click the rendered R.2 table, inspect destinations and render necessary prerequisite figures."""
import hashlib,json,pathlib,urllib.parse,urllib.request
from playwright.sync_api import sync_playwright
ROOT=pathlib.Path(__file__).resolve().parents[5]
OUT=pathlib.Path(__file__).resolve().parent
URL='http://127.0.0.1:8788/course.html#R.2'
result={'preview':URL,'initial_failure':{'command':'python: playwright.chromium.launch(headless=True)','exit_code':1,'stderr':'BrowserType.launch: Executable does not exist at /home/agent/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell; no browser installed by this reviewer. Corrected by using existing /usr/bin/chromium.'},'routes':[],'figures':[]}
with sync_playwright() as p:
    b=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
    result['browser']=b.version
    page=b.new_page(viewport={'width':1400,'height':1000},device_scale_factor=1)
    page.set_default_timeout(10000)
    page.goto(URL,wait_until='domcontentloaded')
    def mark_table():
        page.locator('[id="R.2"]').evaluate('h=>{for(let n=h.nextElementSibling;n&&n.tagName!=="H2";n=n.nextElementSibling){if(n.tagName==="TABLE")n.dataset.r2="1";for(let t of n.querySelectorAll("table"))t.dataset.r2="1";}}')
    mark_table()
    anchors=page.locator('table[data-r2] a').evaluate_all('es=>es.map(e=>({text:e.innerText,href:e.href}))')
    page.locator('table[data-r2]').screenshot(path=str(OUT/'r2-route-table.png'))
    unique={a['href']:i for i,a in reversed(list(enumerate(anchors)))}
    for href,i in sorted(unique.items(),key=lambda x:x[1]):
        if i:page.goto(URL,wait_until='domcontentloaded');mark_table()
        page.locator('table[data-r2] a').nth(i).click()
        page.wait_for_load_state('domcontentloaded')
        ident=urllib.parse.urlsplit(href).fragment or pathlib.PurePosixPath(urllib.parse.urlsplit(href).path).stem
        target=page.locator(f'[id="{ident}"]')
        headings=page.locator('h1,h2').all_text_contents()
        intended=target.count()==1 and target.is_visible() if urllib.parse.urlsplit(href).fragment else any(h.startswith(ident+' ') for h in headings)
        row={'link':anchors[i]['text'],'clicked_href':href,'actual_url':page.url,'target_id':ident,'heading':target.inner_text() if target.count() else headings[0] if headings else '', 'intended_heading_visible':intended}
        if urllib.parse.urlsplit(href).fragment:
            row['anchor_scroll_top']=round(target.bounding_box()['y'],1)
        result['routes'].append(row)
        (OUT/'browser-route-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    page.goto('http://127.0.0.1:8788/17.1.html',wait_until='domcontentloaded')
    links=page.locator('a[href]').evaluate_all('es=>es.filter(e=>/ipynb|colab/.test(e.href)).map(e=>({text:e.innerText,href:e.href}))')
    result['local_notebook_example']={'url':page.url,'heading':page.locator('h1').inner_text(),'notebook_colab_links':links}
    with page.expect_download() as event:
        page.get_by_role('link',name='下載本節 .ipynb',exact=True).click()
    download=event.value
    downloaded=OUT/'downloaded17.1.ipynb'
    download.save_as(str(downloaded))
    data=downloaded.read_bytes()
    nb=json.loads(data)
    result['notebook_content_check']={'download_url':download.url,'suggested_filename':download.suggested_filename,'sha256':hashlib.sha256(data).hexdigest(),'lesson_id':nb.get('metadata',{}).get('lesson_id'),'code_cells':sum(c.get('cell_type')=='code' for c in nb['cells']),'bootstrap_present':any(c.get('metadata',{}).get('course_setup') for c in nb['cells']),'parse_succeeded':True,'matches_current_notebook_bytes':data==(ROOT/'notebooks/17/17.1.ipynb').read_bytes()}
    result['public_site_limit']='Separate canonical HTTPS probe received environment tunnel HTTP 503; see public-route-transport.json. It was not a click destination of the R.2 table. This local executed preview, its actual clicks and Notebook download are the current source review scope.'
    for name in ['posttrain_signals','natural_shared_chat','foundations_tensor_axes','rag']:
        path=ROOT/'course/figures'/f'{name}.svg'
        page.goto(path.as_uri(),wait_until='load')
        page.locator('svg').screenshot(path=str(OUT/f'{name}.png'))
        result['figures'].append({'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'render':str((OUT/f'{name}.png').relative_to(ROOT))})
    b.close()
result['route_count']=len(result['routes'])
result['all_routes_correct']=all(r['intended_heading_visible'] for r in result['routes'])
(OUT/'browser-route-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['routes','figures','initial_failure']},ensure_ascii=False,indent=2))
assert result['all_routes_correct']
