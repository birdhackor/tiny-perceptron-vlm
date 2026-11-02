"""Prompted history checks: fixed-sentence, source counts and cached asset hashes only."""
from pathlib import Path
from html.parser import HTMLParser
import collections,copy,datetime,hashlib,importlib.util,json,platform,re,tomllib
ROOT=Path('/workspace/work/omni-book-integrated-v1');BASE=ROOT/'docs/technical-reviews/artifacts/omni-p7-readme-technical';WAVE=ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/wave05';TASK='/root/omni_p7_technical_readme'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def load(p):return json.loads(p.read_text())
def node(v,p):
 for k in p.split('/')[1:] if p else []:v=v[int(k)] if isinstance(v,list) else v[k.replace('~1','/').replace('~0','~')]
 return v
packet_path=WAVE/'technical-readme/history-requests-groups1and7-v1.json';assert sha(packet_path)=='11da9408c772a3c2eed6993dac72d63b19e4e3dee78a83e98a4a62120ac20a92';packet=load(packet_path)
original=WAVE/'technical-readme/original-v1.json';current=WAVE/'technical-readme/current-v1.json';assert sha(original)=='515f1ad20004f8bf9e2019abc2c4e3237819d8305ec6edaaac73548e90af58ba';assert sha(current)=='24fd5045fbbc43e9b3da4e562c26f3902492d46713cc36b12f9fdc6e8ebfde0f'
readme=(ROOT/'README.md').read_text();assert sha(ROOT/'README.md')=='4ab2217e7d40f0991844dec262ef9e27436a06b945aea649cc0094b3fdeedaac';figure=ROOT/'course/figures/rewrite-01-character-ids.svg';assert sha(figure)=='61d45a7fdaef4ce990a51a9cf924ebc661cf41117f7964bd12d3b1c6c4fb02ad'
# Preserve only the six authenticated historical occurrence nodes and two original unavailability nodes.
preserved=[];groups=[];unique_refs={}
for group in packet['groups']:
 un=group['unavailability'];up=ROOT/un['report']['path'];assert sha(up)==un['report']['sha256'];u=node(load(up),un['json_pointer']);assert u['reviewer_task']==group['old_owner'] and u['unavailable'] is True
 rows=[]
 for item in group['items']:
  dr=item['disposition_ref'];dp=ROOT/dr['path'];assert sha(dp)==dr['sha256'];decision=node(load(dp),dr['json_pointer'])
  issuep=ROOT/item['original_report']['path'];assert sha(issuep)==item['original_report']['sha256'];old_issue=node(load(issuep),item['json_pointer'])
  vr=item['historical_fix_verification'];vp=ROOT/vr['report']['path'];assert sha(vp)==vr['report']['sha256'];verification=node(load(vp),vr['json_pointer'])
  assert decision['issue_id']==item['issue_id'] and decision['report']==item['original_report'] and decision['json_pointer']==item['json_pointer']
  assert decision['action']=='fixed' and decision['final_necessity']==item['necessity']=='necessary' and vr in decision['verification'];assert verification['verdict']=='pass'
  assert verification['reviewer_task']==group['old_owner'];assert old_issue.get('necessary') is True or old_issue.get('severity')=='necessary'
  row={'request_identity':copy.deepcopy(item),'original_issue_node':copy.deepcopy(old_issue),'original_disposition_node':copy.deepcopy(decision),'historical_verification_ref':copy.deepcopy(vr),'historical_verification_header':{k:verification[k] for k in ['schema_version','review_stage','lesson_id','source','source_sha256','figure_sha256','verdict','reviewer_task','resolved_issue_ids'] if k in verification},'preserved_original_necessity':decision['final_necessity'],'old_issue_status':old_issue['status']};rows.append(row)
  for rr in [dr,item['original_report'],vr['report']]:unique_refs[(rr['path'],rr['sha256'])]={'path':rr['path'],'sha256':rr['sha256']}
 groups.append({'group_index':group['group_index'],'old_owner':group['old_owner'],'prior_generation':group['prior_generation'],'unavailability':copy.deepcopy(un),'unavailability_original_node':copy.deepcopy(u),'rows':rows})
assert sum(len(x['rows']) for x in groups)==6
# Still-fixed MathJax wording and the exact current staging/browser implementation.
fixed='公式由網站隨附的 MathJax 在瀏覽器排版。';assert fixed in readme and 'MathJax連網排版' not in readme.replace(' ','')
mathconfig=(ROOT/'course/web/mathjax.js').read_text();assert 'window.MathJax' in mathconfig and 'document$.subscribe' in mathconfig and 'typesetPromise()' in mathconfig and 'fetch(' not in mathconfig
config=tomllib.loads((ROOT/'zensical.toml').read_text());scripts=config['project']['extra_javascript'];assert 'javascripts/mathjax.js' in scripts and 'javascripts/mathjax/tex-chtml.js' in scripts;assert all('://' not in x for x in scripts)
spec=importlib.util.spec_from_file_location('history_mathjax_stage',ROOT/'scripts/stage_mathjax.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);manifest=m.validate_manifest(load(ROOT/'course/web/mathjax-assets.json'))
assert manifest['version']=='3.2.2';cache=ROOT/'outputs/mathjax-cache/3.2.2';preview=ROOT/'outputs/omni-P7-preview-source18-v3';staged=preview/'javascripts/mathjax';asset_rows=[]
for item in manifest['files']:
 for folder in [cache,staged]:m.read_asset(m.asset_path(folder,item['path']),item)
 asset_rows.append({'path':item['path'],'bytes':item['bytes'],'sha256':item['sha256'],'cache_and_staged_match':True})
assert (preview/'javascripts/mathjax.js').read_bytes()==(ROOT/'course/web/mathjax.js').read_bytes()
class P(HTMLParser):
 def __init__(self):super().__init__();self.scripts=[]
 def handle_starttag(self,t,attrs):
  if t=='script':
   a=dict(attrs)
   if 'src' in a:self.scripts.append(a['src'])
html=preview/'readme.html';parser=P();parser.feed(html.read_text());math_src=[x for x in parser.scripts if 'mathjax' in x.lower()];assert any(x.endswith('javascripts/mathjax.js') for x in math_src) and any(x.endswith('javascripts/mathjax/tex-chtml.js') for x in math_src);assert all('://' not in x for x in math_src)
# Recount current chapter sections and current exact Notebook paths; no code-cell execution.
index=load(ROOT/'course/lesson-index.json');indexed={x['id'] for x in index};chapter_ids=[]
for p in sorted((ROOT/'course/chapters').glob('*.md')):chapter_ids.extend(re.findall(r'^## ([\dABC]+\.\d+) ',p.read_text(),re.M))
notebooks=list((ROOT/'notebooks').rglob('*.ipynb'));actual_ids={p.stem for p in notebooks};assert len(index)==len(indexed)==len(chapter_ids)==len(notebooks)==331 and indexed==set(chapter_ids)==actual_ids;assert '6.10' in indexed and (ROOT/'notebooks/06/6.10.ipynb').exists()
for row in index:assert (ROOT/row['notebook']).is_file()
front={name:len(re.findall(r'^## [RWTG]\.\d+ ',(ROOT/name).read_text(),re.M)) for name in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']};assert list(front.values())==[4,7,11,4] and sum(front.values())==26;assert '共331個逐課小節' in readme and '合計357個編號小節' in readme
new_ids=['7.26','9.17','9.18','10.12','12.17','12.18','14.19','16.15','16.16','17.16','17.17','19.21','19.22','19.23','19.24','20.14','20.15','20.16'];assert len(new_ids)==18 and set(new_ids)<=indexed
for rr in unique_refs.values():assert sha(ROOT/rr['path'])==rr['sha256']
preserve={'reviewer_task':TASK,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorized_packet':ref(packet_path),'own_original':ref(original),'own_current':ref(current),'original_current_received_at':'2026-10-30T21:45:33.132289+00:00','mode':'Prompted historical comparison after true independent original/current receipts; not independent rediscovery.','groups':groups,'immutable_original_file_refs_checked':list(unique_refs.values()),'preservation':'Existing original disposition/issue/verification/unavailability file bytes are not rewritten. Their fullSHA and exact JSON pointers bind copied nodes; all issue grades, status strings, old verdicts and verification identities remain as originally stored.'}
out=BASE/'historical-original-nodes-preservation-v1.json';assert not out.exists();out.write_text(json.dumps(preserve,ensure_ascii=False,indent=2)+'\n')
result={'reviewer_task':TASK,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':'python docs/technical-reviews/artifacts/omni-p7-readme-technical/historical-current-check-v1.py','environment':{'python':platform.python_version(),'platform':platform.platform(),'device':'CPU; stdlibJSON/TOML/hash/local asset inspection only'},'source_sha256':sha(ROOT/'README.md'),'figure_sha256':{'course/figures/rewrite-01-character-ids.svg':sha(figure)},'groups':{'1':{'occurrences':5,'old_statuses':[x['old_issue_status'] for x in groups[0]['rows']],'fixed_current_wording':fixed,'MathJax_version':manifest['version'],'MathJax_source_revision':manifest['revision'],'bundled_asset_count':len(asset_rows),'bundled_asset_bytes':sum(x['bytes'] for x in asset_rows),'cached_and_staged_all_SHA_match':True,'browser_typesetPromise_callback_present':True,'configured_script_paths':scripts,'asset_rows':asset_rows,'output_observation_only':{'preview':'outputs/omni-P7-preview-source18-v3','HTML_sha256':sha(html),'mathjax_script_src':math_src,'scope':'Read-only local output corroboration, not a stable implementation source, browser runtime reproduction, public deployment or offline-full-site guarantee.'}},'7':{'occurrences':1,'old_original_stale_counts':[312,338],'old_verified_fixed_counts':[313,339],'current_chapter_source_sections':len(chapter_ids),'current_index':len(index),'current_Notebooks':len(notebooks),'sets_exactly_equal':True,'lesson6_10_retained':True,'front_counts':front,'front_total':sum(front.values()),'current_total':len(index)+sum(front.values()),'current_new_eighteen_ids':new_ids,'interpretation':'The old repair was inventory consistency, not a requirement to keep313/339 forever. Current331+26=357 reflects the18 additional lessons; no off-by-one regression.'}},'original_history_preservation':ref(out),'result':'PASS: both underlying fixes remain true for the six distinct authenticated occurrences; no original/current/report/source bytes modified; no model, teacher, training, kernel, workflow, paid compute or global publication gate executed.'}
output=BASE/'historical-current-check-v1.json';assert not output.exists();output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'created_at':result['created_at'],'group1_assets':len(asset_rows),'group1_bytes':sum(x['bytes'] for x in asset_rows),'group7_counts':[len(index),sum(front.values()),len(index)+sum(front.values())],'result':result['result']},ensure_ascii=False,indent=2))
