from pathlib import Path
from playwright.sync_api import sync_playwright
from hashlib import sha256
from urllib.request import urlopen
import json
base=Path(__file__).parent; out=base/'browser'; records=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    for name,label,visible_marker in [('experiment-source','正式實驗入口','教材實驗與證據'),('gpu-result','原始結果','resumed_from'),('selection-source','驗證決定','selected_variant')]:
        page.goto('http://127.0.0.1:8786/validation.html',wait_until='networkidle')
        a=page.locator('article a').filter(has_text=label).first
        href=a.get_attribute('href')
        a.click(timeout=15000)
        page.wait_for_function('(s)=>document.body.innerText.includes(s)',arg=visible_marker,timeout=18000)
        body=page.locator('body').inner_text()
        (out/(name+'-stable-external.txt')).write_text(body,encoding='utf-8')
        (out/(name+'-stable-external.html')).write_text(page.content(),encoding='utf-8')
        page.screenshot(path=str(out/(name+'-stable-external.png')),full_page=True)
        records.append({'name':name,'actual_visible_label':label,'href':href,'actual_url':page.url,'title':page.title(),'wait_condition':'Visible body includes '+visible_marker,'read_scope':'complete rendered file content within visible GitHub destination; surrounding GitHub chrome incidental','body_sha256':sha256(body.encode()).hexdigest()})
        print('STABLE DESTINATION '+name+'\n'+body+'\n')
    page.goto('http://127.0.0.1:8786/20.12.html',wait_until='networkidle')
    a=page.locator('article a').filter(has_text='在 Colab 動手做').first
    href=a.get_attribute('href'); rec={'name':'colab-destination','from_url':page.url,'actual_visible_label':a.inner_text(),'href':href}
    try:
        a.click(timeout=15000)
        page.wait_for_load_state('domcontentloaded',timeout=18000)
        body=page.locator('body').inner_text()
        (out/'colab-destination.txt').write_text(body,encoding='utf-8')
        page.screenshot(path=str(out/'colab-destination.png'))
        rec.update({'actual_url':page.url,'title':page.title(),'body':body[:1600],'read_scope':'entry destination only; no login or execution'})
    except Exception as e:
        rec.update({'actual_url':page.url,'error':str(e),'read_scope':'browser navigation attempt only'})
    records.append(rec)
    browser.close()
svg=urlopen('http://127.0.0.1:8786/figures/natural-v4-asr-two-routes.svg',timeout=10).read()
(out/'asr-two-routes-preview.svg').write_bytes(svg)
records.append({'name':'preview-original-svg-byte-comparison','original_path':'course/figures/natural-v4-asr-two-routes.svg','url':'http://127.0.0.1:8786/figures/natural-v4-asr-two-routes.svg','original_sha256':sha256(Path('course/figures/natural-v4-asr-two-routes.svg').read_bytes()).hexdigest(),'preview_sha256':sha256(svg).hexdigest(),'equal_bytes':svg==Path('course/figures/natural-v4-asr-two-routes.svg').read_bytes()})
(out/'stable-external-records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
for rel in ['docs/gpu-smoke-result.json','docs/natural-assistant/v4/selection.json']:
    d=base/'snapshots'/rel;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(Path(rel).read_bytes())
print(json.dumps(records,ensure_ascii=False,indent=2))
