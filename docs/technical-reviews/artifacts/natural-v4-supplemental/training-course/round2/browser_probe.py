from pathlib import Path
import json,hashlib
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[6];D=Path(__file__).parent;BASE='http://127.0.0.1:8784/';figures=json.loads((D.parent/'prerequisite-receipts.json').read_text())['figures'];receipt={'base':BASE,'preview_revision':'f43b393431d4908f0ea38a2375f8cae9c4320d12','browser_events':[],'figure_renders':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox']);page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
 page.goto(BASE+'training.html#T.8',wait_until='networkidle');page.locator('[id="T.8"]').scroll_into_view_if_needed();page.screenshot(path=str(D/'browser-training-T8.png'))
 receipt['browser_events'].append({'action':'open complete training.html#T.8, scroll actualheading and capture','url':page.url,'heading':page.locator('[id="T.8"]').inner_text(),'screenshot':'browser-training-T8.png'})
 scope=page.locator('[id="T.8"]').evaluate("e=>{let n=e.nextElementSibling,out=[e.innerText];while(n&&n.tagName!=='H2'){out.push(n.innerText);n=n.nextElementSibling;}return out.join('\\n');}");(D/'browser-T8-DOM.txt').write_text(scope);receipt['browser_events'].append({'action':'actualT.8DOMuntilnextH2','dom_sha256':hashlib.sha256(scope.encode()).hexdigest(),'chars':len(scope)})
 link=page.get_by_role('link',name='2.14的max_memory_allocated文件',exact=True);link.scroll_into_view_if_needed();page.screenshot(path=str(D/'browser-training-memory-link.png'));receipt['browser_events'].append({'action':'view current literal officiallink and memory limits paragraph','url':page.url,'href':link.get_attribute('href'),'visible_label':link.inner_text(),'screenshot':'browser-training-memory-link.png'})
 internal=page.get_by_role('link',name='本節入口',exact=True);internal.scroll_into_view_if_needed();href=internal.get_attribute('href');internal.click();receipt['browser_events'].append({'action':'click actual self-anchor 本節入口 fromT.8','href':href,'url':page.url,'heading':page.locator('[id="T.8"]').inner_text()})
 t4=page.get_by_role('link',name='T.4',exact=True).first;href=t4.get_attribute('href');t4.click();receipt['browser_events'].append({'action':'click actual T.4 link in complete trainingpage','href':href,'url':page.url,'heading':page.locator('[id="T.4"]').inner_text()});page.screenshot(path=str(D/'browser-training-T4.png'))
 for figure in figures:
  raw=(ROOT/figure).read_bytes();url=BASE+'figures/'+Path(figure).name;page.goto(url,wait_until='load');svg=page.locator('svg');name=Path(figure).stem+'.png';svg.screenshot(path=str(D/name));receipt['figure_renders'].append({'source':figure,'source_sha256':hashlib.sha256(raw).hexdigest(),'url':url,'render':name,'render_sha256':hashlib.sha256((D/name).read_bytes()).hexdigest(),'svg_box':svg.bounding_box(),'renderer':'Chromium viaPlaywright','personal_view':'pending'})
 browser.close()
(D/'browser-render-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'events':receipt['browser_events'],'figures':len(figures)},ensure_ascii=False,indent=2))
