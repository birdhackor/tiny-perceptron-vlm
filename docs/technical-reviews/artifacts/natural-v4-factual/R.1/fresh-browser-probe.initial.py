"""Actual Chromium navigation/download/Colab-target probe for R.1."""
from pathlib import Path
import hashlib
import json
import platform
from playwright.sync_api import sync_playwright

ROOT=Path.cwd()
ART=ROOT/'docs/technical-reviews/artifacts/natural-v4-factual/R.1'
RAW=ROOT/'outputs/natural-v4/factual-research/R.1'
BASE='http://127.0.0.1:8788/'
receipt={'steps':[],'colab_runtime_execution':False}

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    context=browser.new_context(viewport={'width':1365,'height':1000},accept_downloads=True)
    page=context.new_page()
    receipt['environment']={'python':platform.python_version(),'browser':browser.version,'viewport':'1365x1000','device':'CPU browser rendering'}
    response=page.goto(BASE+'course.html#R.1',wait_until='networkidle')
    assert response.status == 200
    assert page.locator('[id="R.1"]').count()==1
    page.locator('[id="R.1"]').scroll_into_view_if_needed()
    page.screenshot(path=str(ART/'fresh-R.1-browser.png'))
    receipt['steps'].append({'action':'Read R.1 in executed preview','url':page.url,'status':response.status,
        'heading':page.locator('[id="R.1"]').inner_text(),'warmup_navigation_entries':page.locator('nav a[href="first-steps.html"]').count()})
    page.get_by_role('link',name='第一節：文字怎麼變成數字？',exact=True).click()
    page.wait_for_url('**/1.1.html')
    page.wait_for_load_state('networkidle')
    assert page.locator('[id="1.1"]').count()==1
    assert page.locator('.output pre').inner_text()=="['。', '狗', '看', '貓', '，']\n[3, 2, 1, 4, 1, 2, 3, 0]\n貓看狗，狗看貓。\n"
    receipt['steps'].append({'action':'Click R.1 first-lesson link','url':page.url,'heading':page.locator('[id="1.1"]').inner_text(),
        'visible_code':page.locator('.highlight').first.inner_text(),'visible_actual_output':page.locator('.output pre').inner_text(),
        'output_label':page.locator('.output-label').inner_text(),'warmup_navigation_entries':page.locator('nav a[href="first-steps.html"]').count()})
    figure=page.locator('img[src="figures/character_ids.svg"]')
    figure.scroll_into_view_if_needed()
    figure.screenshot(path=str(ART/'fresh-character-ids-render.png'))
    receipt['figure']={'source':'course/figures/character_ids.svg','sha256':hashlib.sha256((ROOT/'course/figures/character_ids.svg').read_bytes()).hexdigest(),
        'render':'docs/technical-reviews/artifacts/natural-v4-factual/R.1/fresh-character-ids-render.png','natural_width':figure.evaluate('(e)=>e.naturalWidth'),'natural_height':figure.evaluate('(e)=>e.naturalHeight')}
    downloaded_link=page.get_by_role('link',name='下載本節 .ipynb',exact=True)
    receipt['download_href']=downloaded_link.get_attribute('href')
    with page.expect_download() as d:
        downloaded_link.click()
    download=d.value
    download.save_as(str(RAW/'browser-downloaded-1.1.ipynb'))
    b=(RAW/'browser-downloaded-1.1.ipynb').read_bytes()
    current=(ROOT/'notebooks/01/1.1.ipynb').read_bytes()
    assert b==current
    receipt['download']={'suggested_filename':download.suggested_filename,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'matches_current_notebook_bytes':True,
        'markdown_cells':sum(c['cell_type']=='markdown' for c in json.loads(b)['cells']),'code_cells':sum(c['cell_type']=='code' for c in json.loads(b)['cells'])}
    colab_link=page.get_by_role('link',name='在 Colab 動手做 ↗',exact=True)
    colab_url=colab_link.get_attribute('href')
    receipt['colab_target']=colab_url
    with context.expect_page() as popup:
        colab_link.click()
    colab=popup.value
    try:
        colab.wait_for_load_state('domcontentloaded',timeout=45000)
        colab.wait_for_timeout(2500)
        receipt['colab_open']={'url':colab.url,'title':colab.title(),'body_text_excerpt':colab.locator('body').inner_text(timeout=15000)[:5000]}
        colab.screenshot(path=str(RAW/'colab-target.png'))
    except Exception as e:
        receipt['colab_open']={'url':colab.url,'error':str(e)}
    page.goto(BASE+'course.html#R.1',wait_until='networkidle')
    page.get_by_role('link',name='暖身 W.2',exact=True).click()
    page.wait_for_url('**/first-steps.html#W.2')
    assert page.locator('[id="W.2"]').count()==1
    receipt['steps'].append({'action':'Click R.1 W.2 prerequisite','url':page.url,'heading':page.locator('[id="W.2"]').inner_text(),
        'warmup_navigation_entries':page.locator('nav a[href="first-steps.html"]').count()})
    page.go_back(wait_until='networkidle')
    assert '#R.1' in page.url
    page.get_by_role('link',name='暖身 W.1',exact=True).click()
    page.wait_for_url('**/first-steps.html#W.1')
    assert page.locator('[id="W.1"]').count()==1
    receipt['steps'].append({'action':'Click R.1 W.1 operation instructions','url':page.url,'heading':page.locator('[id="W.1"]').inner_text(),
        'has_actual_local_notebook_path':'01/1.1.ipynb' in page.locator('article').inner_text()})
    browser.close()

(ART/'fresh-browser-results.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
