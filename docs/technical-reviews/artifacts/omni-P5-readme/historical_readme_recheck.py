"""Prompted historical recheck for request array indices1 and7 only; no NN or source mutation."""
import collections,datetime,functools,hashlib,http.server,importlib.util,json,re,threading
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright
R=Path('/workspace/work/omni-book-integrated-v1');D=R/'docs/technical-reviews/artifacts/omni-P5-readme';RAW=Path('/workspace/work/omni-P5-readme-independent-raw');BASE=R/'docs/course-revision-20261029-omni/reviews/P5-actual';PREVIEW=R/'outputs/omni-P5-preview-v9'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dig(q):
 q=Path(q);return {'path':str(q.relative_to(R)) if q.is_relative_to(R) else str(q),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
def read(q):return json.loads(Path(q).read_text())
def node(d,p):
 for k in p.split('/')[1:]:d=d[int(k)] if isinstance(d,list) else d[k.replace('~1','/').replace('~0','~')]
 return d
request_path=R/'docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json';request=read(request_path);groups={i:request['groups'][i] for i in [1,7]}
assert [(g['page_id'],len(g['items'])) for g in groups.values()]==[('readme',5),('readme',1)]
p={'reviewer_task':'/root/omni_p5_technical_readme','started_at':now(),'kind':'prompted_historical_recheck_after_own_original_and_current_intake','command':'python docs/technical-reviews/artifacts/omni-P5-readme/historical_readme_recheck.py','request':dig(request_path),'request_array_indices':[1,7],'initial_review_separation':'Original/current1 both immutable and actually collected before historical disclosure. Earlier mistaken1-based extraction exposed metadata paths/IDs only of indices0/6; no corresponding reports/dispositions or judgments read, no review performed for them. Correct indices1/7 exclusively examined here.'}
original=BASE/'technical-readme/technical-original-v1.json';full=BASE/'technical-readme/technical-current-v1.json';p['own_original']=dig(original);p['full_review']=dig(full)
assert p['own_original']['sha256']=='d5b9b1e8c5bede654aff3412120e5e016ba860beb295aa246e05c80bc20dc1e9';assert p['full_review']['sha256']=='7f2fb3327cbb2e08bad8b88457ccaa520ebab825ef315706865f04e7a299ef13'
intake_path=BASE/'technical-original-collection-intake-v10.json';intake=read(intake_path);p['actual_prior_collection']={'artifact':dig(intake_path),'rows':[row for row in intake['collection_index'] if row['sha256'] in {p['own_original']['sha256'],p['full_review']['sha256']}]}
assert len(p['actual_prior_collection']['rows'])==2 and all(row['received_at']<p['started_at'] for row in p['actual_prior_collection']['rows'])
f=read(BASE/'source-freeze-v10.json');page=next(x for x in f['pages'] if x['page_id']=='readme');body=(R/'README.md').read_text();p['current_source']=dig(R/'README.md');p['frozen_source']=page;assert p['current_source']['sha256']==page['source_sha256']==read(full)['source_sha256']
p['current_quotes']={'mathjax':'公式由網站隨附的 MathJax 在瀏覽器排版。','counts':'20章與A／B延伸單元及C推理計算必修單元，共331個逐課小節，Notebook由同一份來源產生；閱讀指南、暖身、訓練操作與名詞頁另有26個編號小節，合計357個編號小節。'}
assert all(q in body for q in p['current_quotes'].values())
p['selected_history']=[];unchanged={}
for group_index,g in groups.items():
 row={'group_index':group_index,'page_id':g['page_id'],'old_owner':g['old_owner'],'unavailability':g['unavailability'],'instances':[]}
 u=g['unavailability'];up=R/u['report']['path'];assert dig(up)['sha256']==u['report']['sha256'];unchanged[str(up)]=dig(up)['sha256'];un=node(read(up),u['json_pointer']);row['unavailability_observation']={'reviewer_task':un['reviewer_task'],'unavailable':un['unavailable'],'recorded_at':un['recorded_at'],'followup_response':un['events'][0]['response'],'listing_response':un['events'][1]['response']}
 for item in g['items']:
  dr=item['disposition_ref'];dp=R/dr['path'];assert dig(dp)['sha256']==dr['sha256'];unchanged[str(dp)]=dig(dp)['sha256'];dis=node(read(dp),dr['json_pointer']);op=R/item['original_report']['path'];assert dig(op)['sha256']==item['original_report']['sha256'];unchanged[str(op)]=dig(op)['sha256'];issue=node(read(op),item['json_pointer']);hv=item['historical_fix_verification'];hp=R/hv['report']['path'];assert dig(hp)['sha256']==hv['report']['sha256'];unchanged[str(hp)]=dig(hp)['sha256']
  assert dis['final_necessity']==item['necessity']=='necessary' and dis['action']=='fixed' and dis['report']==item['original_report'] and dis['json_pointer']==item['json_pointer'] and hv in dis['verification']
  row['instances'].append({'requested_item':item,'original_issue_node':issue,'old_action':dis['action'],'old_final_necessity':dis['final_necessity'],'old_verification_owner':read(hp)['reviewer_task'],'old_source_sha256':read(hp)['source_sha256'],'interpretation':'historical identity/provenance only; old pass is not current technical evidence'})
 p['selected_history'].append(row)
# Fresh current-count recomputation, not an old count constant.
index=read(R/'course/lesson-index.json');indexed={x['id'] for x in index};notebooks={q.stem for q in (R/'notebooks').rglob('*.ipynb')};source_ids={i for q in (R/'course/chapters').glob('*.md') for i in re.findall(r'^## ([A-Z0-9]+\.\d+) ',q.read_text(),re.M)};guides={name:len(re.findall(r'^## ([A-Z0-9]+\.\d+) ',(R/name).read_text(),re.M)) for name in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']}
p['count_recheck']={'lesson_index':dig(R/'course/lesson-index.json'),'index_rows':len(index),'unique_index_ids':len(indexed),'notebook_files':len(notebooks),'chapter_source_sections':len(source_ids),'source_index_notebook_sets_equal':indexed==notebooks==source_ids,'guide_counts':guides,'guide_total':sum(guides.values()),'total_numbered':len(index)+sum(guides.values()),'quote_from_current':p['current_quotes']['counts'],'scope':'331current lessons +26guide sections =357; old313/339 numbers are not blindly reused.'}
assert indexed==notebooks==source_ids and len(index)==331 and sum(guides.values())==26 and p['count_recheck']['total_numbered']==357
# Source, manifest, and every current preview MathJax asset, without staging/downloading anything.
spec=importlib.util.spec_from_file_location('historical_mathjax_stage',R/'scripts/stage_mathjax.py');stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage);manifest=stage.validate_manifest(read(R/'course/web/mathjax-assets.json'));assets=[]
for item in manifest['files']:
 q=PREVIEW/'javascripts/mathjax'/item['path'];raw=q.read_bytes();stage.verify(raw,item);assets.append({'path':item['path'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
p['mathjax_assets']={'version':manifest['version'],'revision':manifest['revision'],'manifest':dig(R/'course/web/mathjax-assets.json'),'files_verified':len(assets),'total_bytes':sum(x['bytes'] for x in assets),'all_files_sha_and_size_match':True,'tex_chtml':next(x for x in assets if x['path']=='tex-chtml.js'),'script_config':dig(R/'course/web/mathjax.js'),'stage_source':dig(R/'scripts/stage_mathjax.py'),'export_source':dig(R/'scripts/export_course.py'),'site_config':dig(R/'zensical.toml'),'scope':'Read existing preview assets only; did not call prepare_assets/stage_assets/build or publish.'}
class H(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(H,directory=str(PREVIEW)));threading.Thread(target=server.serve_forever,daemon=True).start();port=server.server_address[1]
p['browser']={'local_url':f'http://127.0.0.1:{port}/readme.html','attempted_external_requests':[],'request_urls':[]}
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path='/workspace/work/browsers/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox']);p['browser']['version']=browser.version;ctx=browser.new_context(viewport={'width':1200,'height':900})
  def local_only(route):
   u=route.request.url
   if urlsplit(u).hostname in {'127.0.0.1','localhost'}:route.continue_()
   else:p['browser']['attempted_external_requests'].append(u);route.abort()
  ctx.route('**/*',local_only);page_b=ctx.new_page();page_b.on('request',lambda req:p['browser']['request_urls'].append(req.url));page_b.goto(p['browser']['local_url'],wait_until='networkidle');page_b.wait_for_function('window.MathJax && !!window.MathJax.typesetPromise');page_b.evaluate('MathJax.startup.promise')
  p['browser']['script_srcs']=page_b.locator('script[src]').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("src"))')
  p['browser']['artificial_formula']=page_b.evaluate(r'''async()=>{const n=document.createElement('div');n.className='arithmatex';n.id='independent-history-math';n.textContent='\\(x^2+1\\)'.replaceAll('\\\\','\\');document.querySelector('article').appendChild(n);await MathJax.typesetPromise([n]);return {MathJax_version:MathJax.version,typesetPromise_type:typeof MathJax.typesetPromise,generated_containers:n.querySelectorAll('mjx-container').length,output_text:n.innerText,html:n.innerHTML.slice(0,500)};}''')
  assert p['browser']['artificial_formula']['generated_containers']==1
  shot=RAW/'historical-mathjax-browser-formula.png';page_b.locator('#independent-history-math').screenshot(path=str(shot));p['browser']['screenshot']=dig(shot);browser.close()
finally:server.shutdown()
assert any(x.endswith('javascripts/mathjax/tex-chtml.js') for x in p['browser']['script_srcs']) and any(x.endswith('javascripts/mathjax.js') for x in p['browser']['script_srcs'])
p['browser']['scope']='Added x^2+1 only to ephemeral browser DOM to verify actual typesetPromise using packaged3.2.2. README source untouched. Only localhost assets permitted; does not claim all pages/expressions or OS offline operation.'
# Small exact primary-source excerpts. Full third-party docs stay outside Git.
p['official_docs']=[]
for name,file,url,term in [('typeset','mathjaxTypeset-v3.2-a39fc2d.rst','https://raw.githubusercontent.com/mathjax/MathJax-docs/a39fc2d1f7ef82477636c1aeb9dd8d8fc3eb3fe4/web/typeset.rst','Typesetting Math in a Web Page'),('hosting','mathjaxHosting-v3.2-a39fc2d.rst','https://raw.githubusercontent.com/mathjax/MathJax-docs/a39fc2d1f7ef82477636c1aeb9dd8d8fc3eb3fe4/web/hosting.rst','We recommend using a CDN')]:
 q=RAW/file;s=q.read_text();i=s.find(term);p['official_docs'].append({'name':name,'url':url,'version':'MathJax-docs v3.2 branch at a39fc2d1f7ef82477636c1aeb9dd8d8fc3eb3fe4','accessed_on':'2026-10-30','raw_receipt':dig(q),'locator':term,'excerpt':s[i:i+1300]})
p['network_access_note']='docs.mathjax.org v3.2 URLs returned403; no source claims made from denied HTML. Read publisher original v3.2 docs via GitHub at exact resolved branch commit instead. Initial guessed3.2-latest ref404 discarded; actualv3.2 branch commit used.'
p['source_references']=[dig(R/name) for name in ['README.md','scripts/stage_mathjax.py','scripts/export_course.py','course/web/mathjax.js','course/web/mathjax-assets.json','zensical.toml','course/lesson-index.json','docs/review-tools/historical_fix_rechecks.py']]
p['old_bytes_unchanged']=all(dig(Path(q))['sha256']==h for q,h in unchanged.items());p['own_full_original_current_unchanged']=(dig(original)==p['own_original'] and dig(full)==p['full_review']);p['completed_at']=now();assert p['old_bytes_unchanged'] and p['own_full_original_current_unchanged']
with (D/'historical-readme-groups-1-7-proof-v1.json').open('x') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'proof':dig(D/'historical-readme-groups-1-7-proof-v1.json'),'group_instances':[len(g['instances']) for g in p['selected_history']],'counts':p['count_recheck'],'mathjax_assets':{k:v for k,v in p['mathjax_assets'].items() if k in ['version','revision','files_verified','total_bytes','all_files_sha_and_size_match']},'browser_formula':p['browser']['artificial_formula'],'external_requests':p['browser']['attempted_external_requests'],'old_bytes_unchanged':p['old_bytes_unchanged']},ensure_ascii=False,indent=2))
