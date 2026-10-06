"""Actual current local site, exact T.10 sibling scope, no other section body."""
import hashlib
import importlib.metadata
import json
import pathlib
import re
import sys
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin

ROOT=pathlib.Path(__file__).resolve().parents[4]
OUT=pathlib.Path(__file__).resolve().parent
fences=re.findall(rb'```bash\n(.*?)```',OUT.joinpath('current/T.10.md').read_bytes(),re.S)
records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    for label,width,height in [('desktop',1280,800),('mobile',390,844)]:
        page=browser.new_page(viewport={'width':width,'height':height},device_scale_factor=1)
        response=page.goto('http://127.0.0.1:8765/training.html',wait_until='networkidle')
        assert response.status==200
        scope=page.evaluate('''() => {
          const h=document.getElementById('T.10');
          if (!h || h.tagName!=='H2') throw new Error('exact own H2 missing');
          const nodes=[h];let n=h.nextElementSibling;
          while(n && n.tagName!=='H2'){nodes.push(n);n=n.nextElementSibling;}
          const html=nodes.map(x=>x.outerHTML).join(String.fromCharCode(10));
          const code=nodes.flatMap(x=>Array.from(x.querySelectorAll('pre code'))).map(x=>x.textContent);
          const links=nodes.flatMap(x=>Array.from(x.querySelectorAll('a'))).map(x=>x.getAttribute('href'));
          const keep=new Set(nodes);
          for(const child of Array.from(h.parentElement.children)) if(!keep.has(child)) child.remove();
          for(const d of h.parentElement.querySelectorAll('details')) d.open=true;
          return {html,code,links,next_h2_id:n ? n.id : null};
        }''')
        assert [x.encode() for x in scope['code']]==fences
        assert scope['next_h2_id']=='T.11'
        actual_links={urljoin(page.url,x) for x in scope['links']}
        assert all(urljoin(page.url,href) in actual_links for href in ['training.html#T.4','training.html#T.6','12.12.html'])
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(OUT/f'site-{label}-own-scope.png'),full_page=True)
        details=page.locator('details').filter(has_text='原始資料、教師與多模態路線').last
        details.scroll_into_view_if_needed()
        page.evaluate('''() => {const ds=Array.from(document.querySelectorAll('details')).find(d=>d.textContent.includes('原始資料、教師與多模態路線'));window.scrollTo(0,ds.getBoundingClientRect().top+window.scrollY-20);}''')
        page.screenshot(path=str(OUT/f'site-{label}-new-recipe-viewport.png'))
        overflow=page.evaluate('''() => ({viewport:innerWidth,body:document.body.scrollWidth,
           code: Array.from(document.querySelectorAll('article pre')).map(p=>({width:p.clientWidth,scroll:p.scrollWidth,overflow:getComputedStyle(p).overflowX}))})''')
        OUT.joinpath(f'site-{label}-own-dom.html').write_text(scope['html'])
        records.append({'viewport':[width,height],'response_status':response.status,'URL':page.url,
            'scope':'Actual loaded page pruned to exact T.10 H2 siblings after extracting/checking original DOM; details expanded; head CSS/layout retained. Screenshots depict only own body, not unmodified entire chapter.',
            'own_dom_sha256':hashlib.sha256(scope['html'].encode()).hexdigest(),'code_matches_source':True,
            'next_H2_id':scope['next_h2_id'],'necessary_links_checked':True,'layout':overflow,
            'screenshots':[f'site-{label}-own-scope.png',f'site-{label}-new-recipe-viewport.png']})
        page.close()
    version=browser.version
    browser.close()
result={'python':sys.version,'python_executable':sys.executable,'playwright':importlib.metadata.version('playwright'),
        'browser':'/usr/bin/chromium','chromium':version,'device':'CPU/browser','screens':records,
        'scope':'Visual check of exact current T.10 only. No other section report/body opened.'}
OUT.joinpath('site-visual-execution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
