"""Post-original independent check of five prior MathJax fix records."""
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from playwright.sync_api import sync_playwright

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=Path('/workspace/work/tokenizer-dynamic-20261027/readme-callback')
A=ROOT/'docs/technical-reviews/artifacts/tokenizer-20261027/readme-callback'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_bytes())
started=datetime.now(timezone.utc).isoformat()
original=ROOT/'docs/course-review-20261027-tokenizer/reviews/technical-readme-callback/technical-original-01.json'
original_identity={'path':str(original.relative_to(ROOT)),'sha256':sha(original),'created_at':load(original)['created_at']}
disposition_path=ROOT/'docs/course-review-20261019/dispositions-continuity-01.json'
dispositions=load(disposition_path)
records=[]
for index in range(196,201):
    item=dispositions['items'][index]
    old=load(ROOT/item['report']['path'])
    verification=item['verification'][0]
    closed=load(ROOT/verification['report']['path'])
    records.append({'index':index,'issue_id':item['issue_id'],'original_report':item['report'],'original_sha_matches':sha(ROOT/item['report']['path'])==item['report']['sha256'],'original_issue':old['issues'][0],'original_json_pointer':item['json_pointer'],'historical_fix_verification':verification,'fix_sha_matches':sha(ROOT/verification['report']['path'])==verification['report']['sha256'],'historical_fix_verdict':closed['verdict'],'historical_fix_issue':closed['issues'][0]})

url='https://docs.mathjax.org/en/v3.2/web/hosting.html'
raw=urlopen(Request(url,headers={'User-Agent':'README independent MathJax fix check'}),timeout=25).read()
raw_path=OUT/'mathjax-hosting-v3.2.html';raw_path.write_bytes(raw)
plain=re.sub(r'<(script|style)\b[^>]*>.*?</\1>','',raw.decode(),flags=re.S)
plain=re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',plain))
excerpts=[]
for term in ['locally','server','fonts']:
    found=re.search(re.escape(term),plain,re.I)
    if found: excerpts.append({'term':term,'text':plain[max(0,found.start()-150):found.end()+550]})
official={'url':url,'version':'MathJax documentation v3.2','accessed_on':'2026-10-27','raw_path':str(raw_path),'raw_sha256':sha(raw_path),'excerpts':excerpts}
manifest=load(ROOT/'course/web/mathjax-assets.json')
site=ROOT/'outputs/tokenizer-20261027/site'
asset_checks=[]
for item in manifest['files']:
    p=site/'javascripts/mathjax'/item['path']
    asset_checks.append({'path':item['path'],'bytes_match':p.stat().st_size==item['bytes'],'sha256_match':sha(p)==item['sha256']})
requests=[];blocked=[];errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1000,'height':800},reduced_motion='reduce')
    def route(r):
        u=r.request.url
        if urlsplit(u).hostname not in ['127.0.0.1','localhost'] and not u.startswith(('data:','blob:')):
            blocked.append(u);r.abort()
        else:r.continue_()
    page.route('**/*',route)
    page.on('request',lambda r:requests.append(r.url))
    page.on('pageerror',lambda e:errors.append(str(e)))
    response=page.goto('http://127.0.0.1:8000/readme.html',wait_until='networkidle')
    page.wait_for_function('window.MathJax && window.MathJax.typesetPromise && window.MathJax.startup')
    result=page.evaluate('''async () => {
      await MathJax.startup.promise;
      const e=document.createElement('div'); e.className='arithmatex'; e.id='reviewer-mathjax-probe';
      e.style='padding:24px;font-size:30px;background:white;color:black;width:500px';
      e.textContent='\\\\(x+y=3\\\\)'; document.body.prepend(e);
      await MathJax.typesetPromise([e]);
      return {version:MathJax.version,containers:e.querySelectorAll('mjx-container').length,renderedText:e.innerText,innerHTML:e.innerHTML};
    }''')
    image=OUT/'mathjax-browser-probe-01.png'
    page.locator('#reviewer-mathjax-probe').screenshot(path=str(image))
    scripts=page.locator('script[src]').evaluate_all('(nodes)=>nodes.map(n=>n.src)')
    browser.close()
inspect=load(A/'inspect-readme-01.json')
cuda=[r for r in inspect['historical_gpu_records'] if r['device']=='cuda']
cpu=[r for r in inspect['historical_gpu_records'] if r['device']=='cpu']
receipt={'schema_version':1,'owner':'/root/tech_readme_callback_20261027','started_at':started,'created_at':datetime.now(timezone.utc).isoformat(),'original_saved_before_lookup':original_identity,'scope':'Only five prior fix identity records checked after own original full review. All five refer to one MathJax issue at successive old report versions. Browser ephemeral equation was added only to test client typesetting; README files unchanged. Current full report also records a reviewer-only GPU aggregation correction; saved original evidence retained.',
 'current_source_sha256':sha(ROOT/'README.md'),'current_figure_sha256':{'course/figures/rewrite-01-character-ids.svg':sha(ROOT/'course/figures/rewrite-01-character-ids.svg')},'current_sentence_present':'公式由網站隨附的 MathJax 在瀏覽器排版。' in (ROOT/'README.md').read_text(),'old_sentence_absent':'公式由 MathJax 連網排版' not in (ROOT/'README.md').read_text(),'disposition_sha256':sha(disposition_path),'records':records,'official_original':official,'current_bindings':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/stage_mathjax.py',ROOT/'course/web/mathjax.js',ROOT/'course/web/mathjax-assets.json',ROOT/'scripts/export_course.py',ROOT/'zensical.toml']},'site_asset_checks':{'total':len(asset_checks),'all_match':all(c['bytes_match'] and c['sha256_match'] for c in asset_checks),'files':asset_checks},'browser':{'url':'http://127.0.0.1:8000/readme.html','http_status':response.status,'external_hosts_blocked':True,'blocked_requests':blocked,'all_request_hosts':sorted({str(urlsplit(u).hostname) for u in requests}),'script_src':scripts,'mathjax_requests':[u for u in requests if 'mathjax' in u.lower()],'page_errors':errors,'typeset_result':result,'png':str(image),'png_sha256':sha(image)},'reviewer_summary_correction':{'original_observation_to_correct':'r15 observed/a-inspect artifact description said all30 formal report environments were L4','actual':{'formal_results':len(inspect['historical_gpu_records']),'cuda_reports':len(cuda),'cpu_reports':len(cpu),'cuda_gpu_names':sorted({str(r['gpu']) for r in cuda}),'cpu_experiment_ids':[r['id'] for r in cpu],'all_torch_versions':sorted({str(r['torch_version']) for r in inspect['historical_gpu_records']})},'effect':'README says formal GPU training used L4, which remains supported by all29 actual CUDA reports; simple_models used CPU. Original report immutable; current narrows summary precisely.'},'environment':{'python':platform.python_version(),'browser':'Chromium via Playwright','device':'CPU browser'}}
(A/'historical-probe-readme-01.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'created_at':receipt['created_at'],'records':len(records),'all_assets_match':receipt['site_asset_checks']['all_match'],'browser':{'status':response.status,'hosts':receipt['browser']['all_request_hosts'],'blocked':blocked,'version':result['version'],'containers':result['containers'],'page_errors':errors},'reviewer_correction':receipt['reviewer_summary_correction']['actual'],'official':official},ensure_ascii=False))
