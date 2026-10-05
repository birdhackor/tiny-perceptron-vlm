"""Finish already captured navigation evidence and render figures over the actual preview HTTP route."""
import hashlib,json,pathlib,subprocess
from playwright.sync_api import sync_playwright
ROOT=pathlib.Path(__file__).resolve().parents[5]
OUT=pathlib.Path(__file__).resolve().parent
result=json.loads((OUT/'browser-route-results.json').read_text())
assert len(result['routes'])==41
result['additional_failures']=[{'command':'python browser_routes.py (second attempt)','exit_code':1,'evidence':'browser-command.stderr','error':'Chromium blocks file:// SVG navigation (ERR_BLOCKED_BY_ADMINISTRATOR). All 41 table links and the Notebook download completed before this failure. Fixed figure rendering by using the actual preview HTTP SVG paths.'}]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1400,'height':1000},device_scale_factor=1)
 result['browser']=b.version
 page.goto('http://127.0.0.1:8788/17.1.html',wait_until='domcontentloaded')
 links=page.locator('a[href]').evaluate_all('es=>es.filter(e=>/ipynb|colab/.test(e.href)).map(e=>({text:e.innerText,href:e.href}))')
 result['local_notebook_example']={'url':page.url,'heading':page.locator('h1').inner_text(),'notebook_colab_links':links}
 data=(OUT/'downloaded17.1.ipynb').read_bytes();nb=json.loads(data)
 result['notebook_content_check']={'download_url':next(l['href'] for l in links if 'colab.' not in l['href']),'download_command_origin':'Actual Playwright expect_download and download.save_as in browser_routes.py, before the file:// rendering failure','sha256':hashlib.sha256(data).hexdigest(),'lesson_id':nb['metadata']['lesson_id'],'code_cells':sum(c['cell_type']=='code' for c in nb['cells']),'bootstrap_present':any(c['metadata'].get('course_setup') for c in nb['cells']),'parse_succeeded':True,'matches_current_notebook_bytes':data==(ROOT/'notebooks/17/17.1.ipynb').read_bytes()}
 result['colab_target_git_check']={'command':'git cat-file -e 25ca4bf5f8f8dc0a9e88aa6cd70caeb4c2f419e8:notebooks/17/17.1.ipynb','exit_code':subprocess.run(['git','cat-file','-e','25ca4bf5f8f8dc0a9e88aa6cd70caeb4c2f419e8:notebooks/17/17.1.ipynb'],cwd=ROOT).returncode,'scope':'Confirms exact named source revision/path exists; no Colab kernel launched.'}
 result['figures']=[]
 for name in ['posttrain_signals','natural_shared_chat','foundations_tensor_axes','rag']:
  path=ROOT/'course/figures'/f'{name}.svg'
  response=page.goto('http://127.0.0.1:8788/figures/'+name+'.svg',wait_until='load')
  published=response.body();assert published==path.read_bytes()
  page.locator('svg').screenshot(path=str(OUT/f'{name}.png'))
  result['figures'].append({'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(published).hexdigest(),'actual_render_url':page.url,'render':str((OUT/f'{name}.png').relative_to(ROOT))})
 b.close()
result['route_count']=len(result['routes']);result['all_routes_correct']=all(r['intended_heading_visible'] for r in result['routes'])
result['public_site_limit']='Separately requested canonical HTTPS page returned environment tunnel HTTP 503 in Chromium and urllib; see public-route-transport.json. It was not any R.2 table click destination. Review scope is current local executed preview and actual click semantics.'
(OUT/'browser-route-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['routes','figures','initial_failure']},ensure_ascii=False,indent=2))
assert result['all_routes_correct'] and result['notebook_content_check']['matches_current_notebook_bytes']
