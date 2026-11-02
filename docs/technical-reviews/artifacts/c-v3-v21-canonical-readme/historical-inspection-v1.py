from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib.metadata,json,platform,re,tomllib
from scripts.stage_mathjax import validate_manifest
R=Path('/tmp/tpb').resolve();A=R/'docs/technical-reviews/artifacts/c-v3-v21-canonical-readme';C=R/'docs/course-revision-20261101-c-v3';P=C/'technical-coordination-v21/history-prompts'
ACTOR='/root/c_v3_technical_coord/canonical_readme'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ptr(d,p):
 for k in p.strip('/').split('/') if p else []:d=d[int(k)] if isinstance(d,list) else d[k.replace('~1','/').replace('~0','~')]
 return d
def auth(ref):
 p=R/ref['path'];assert sha(p)==ref['sha256'],ref['path'];return j(p)
def rpath(p):return str(Path(p).relative_to(R))
out={'actor':ACTOR,'started_at':datetime.now(timezone.utc).isoformat(),'physical_root':str(R),'environment':{'python':platform.python_version(),'torch':importlib.metadata.version('torch'),'platform':platform.platform(),'device':'cpu; bounded text/JSON/AST/hash and manifest validation only'},'scope':'Genuine post-full-original/current historical callback after resume. Six authenticated existing fixes only. No whole prior114snapshot replay, no training/model inference/kernels/Windows/website/browser/full product acceptance, no install/network/final/Modal/commit/publication. Prior appendix still_fixed conclusions were not consulted.'}
full_refs={'original':{'path':'docs/course-revision-20261101-c-v3/technical-v21/canonical_readme/original-v1.json','sha256':'da05ffd71e0165e414f4643b11919231e6f40f24d99e0c74dce573dc4984375a'},'current':{'path':'docs/course-revision-20261101-c-v3/technical-v21/canonical_readme/current-v1.json','sha256':'c4abbadbb5f9c77e89197c7e19f2ce12cbc933c4eabc7aaa650c739591774522'}}
for ref in full_refs.values():auth(ref)
current=auth(full_refs['current']);intakes={}
for kind,expected in [('original','3fade71eaaa0ca9e8c3f720197f8a2d2ef83e0218482f2fcb40e8de8d1d3371e'),('current','eb66af66aae2c1ca028f60367d80de87e2b594cb4238a245e47475ee2f8b196e')]:
 p=C/f'technical-coordination-v21/receipts/canonical-readme-{kind}-v1.json';assert sha(p)==expected;d=j(p);assert d['report']==full_refs[kind];intakes[kind]={'path':rpath(p),'sha256':expected,'received_at':d['received_at']}
assert out['started_at']>intakes['current']['received_at'];out['prior_full_reviews']=full_refs;out['actual_intakes']=intakes
freeze_ref={'path':'docs/course-revision-20261101-c-v3/source-freeze-v22.json','sha256':'0f4e9d63bfcb2243234c17c62242ff79cd4ab640a21d1f313f94c99946a93a59'};freeze=auth(freeze_ref);page=next(x for x in freeze['pages'] if x['page_id']=='readme')
assert page['source_sha256']==sha(R/'README.md')==current['source_sha256']=='be92d9828a3bf551fc459348770468ba62f34d390c1e2e66b95e2bd8cee487a5'
assert page['figures_sha256']==current['figure_sha256'] and sha(R/'course/figures/rewrite-01-character-ids.svg')==current['figure_sha256']['course/figures/rewrite-01-character-ids.svg']
out['current_freeze']={**freeze_ref,'readme_entry':page};deps=[]
for s in current['sources']:
 if s['kind']=='repository_code':assert sha(R/s['path'])==s['sha256'];deps.append({'kind':'source','id':s['id'],'path':s['path'],'sha256':s['sha256']})
for a in current['artifacts']:assert sha(R/a['path'])==a['sha256'];deps.append({'kind':'artifact','id':a['id'],'path':a['path'],'sha256':a['sha256']})
out['own_full_current_actual_dependency_identity']={'checked_references':deps,'new_drift':[],'scope':'Only sources/artifacts actually in this actor fullcurrent; no prior114snapshot batch rerun. Source22README bytes unchanged; 19.25format/NB not an input to these six count/MathJax fixes.'}
packets=[]
for filename,expected in [('child-00-group-11-historical-identities.json','44134107f7291baf70fb760fa0dd391e003d23010c8e0eb3dd463ff66c0e939d'),('child-01-group-02-historical-identities.json','da18a926d88496b2d6f19876f389aa8bde8318fdec174d05899deef0fe6fc648')]:
 p=P/filename;assert sha(p)==expected;packet=j(p);auth(packet['prior_active_pointer']);auth(packet['source_chain']);unavail=auth(packet['unavailability']['report']);unavailrow=ptr(unavail,packet['unavailability']['json_pointer']);rows=[]
 for row in packet['historical_identities_only']:
  original=auth(row['original_report']);issue=ptr(original,row['json_pointer']);disposition=ptr(auth(row['disposition_ref']),row['disposition_ref']['json_pointer']);fixed=auth(row['historical_fix_verification']['report']);assert row['historical_fix_verification'] in disposition['verification'];assert disposition['action']=='fixed' and disposition['final_necessity']==row['original_necessity']=='necessary';assert disposition['issue_id']==row['issue_id'] and disposition['report']==row['original_report'] and disposition['json_pointer']==row['json_pointer'];assert original['reviewer_task']==fixed['reviewer_task']==unavailrow['reviewer_task']
  # Hash a prior appendix as an identity only; do not read its recheck verdict/reason.
  previous=row.get('prior_appendix_row')
  if previous:assert sha(R/previous['report']['path'])==previous['report']['sha256']
  rows.append({'identity':row,'original_issue':issue,'accepted_disposition':{k:disposition[k] for k in ['issue_id','final_necessity','action','source_support']},'original_owner':original['reviewer_task'],'original_fix_source_sha256':fixed['source_sha256'],'original_fix_report':row['historical_fix_verification'],'prior_appendix_identity_only':previous,'necessity_kept':disposition['final_necessity']})
 packets.append({'prompt':{'path':rpath(p),'sha256':expected},'source_group_index':packet['source_group_index'],'source_chain':packet['source_chain'],'prior_active_pointer':packet['prior_active_pointer'],'unavailability':packet['unavailability'],'original_unavailability_record':unavailrow,'rows':rows})
out['authenticated_packets']=packets
readme=(R/'README.md').read_text(encoding='utf-8');countline=next(x for x in readme.splitlines() if x.startswith('20章與'))
index=j(R/'course/lesson-index.json');nbpaths={str(x.relative_to(R)) for x in (R/'notebooks').rglob('*.ipynb')};ids={x['id'] for x in index};chapterids=[];chaptercounts={}
for p in sorted((R/'course/chapters').glob('*.md')):
 values=re.findall(r'^##\s+([0-9ABC]+\.\d+)(?:\s|$)',p.read_text(),re.M);chapterids+=values;chaptercounts[rpath(p)]=len(values)
assert len(index)==len(ids)==len(chapterids)==len(set(chapterids))==len(nbpaths)==332 and ids==set(chapterids) and {x['notebook'] for x in index}==nbpaths
extras={}
for name in ['course/README.md','course/first-steps.md','course/training.md','course/glossary.md']:extras[name]=re.findall(r'^##\s+([A-Z]+\.\d+)(?:\s|$)',(R/name).read_text(),re.M)
extra=sum(len(x) for x in extras.values());assert extra==26 and 332+extra==358 and '332' in countline and '26' in countline and '358' in countline
eng=(R/'README_en.md').read_text();engline=next(x for x in eng.splitlines() if '332' in x and '358' in x);assert '26' in engline
countfix=R/'docs/technical-reviews/artifacts/tokenizer-20261027/readme/recheck-output-01.json';countbrowser=R/'docs/technical-reviews/artifacts/tokenizer-20261027/readme/recheck-render-01.json';assert sha(countfix)=='69dc630463fbec5444e6e995faf3de9f8990326aaa2728bb2a93d079c617c5f4' and sha(countbrowser)=='9de056db1591f515f2e5c773c28bb073716a5528da89386fde1158ff730a4628';cf=j(countfix);cb=j(countbrowser)
assert cf['count']['notebooks']==313 and cf['count']['extra']==26 and cf['count']['total']==339 and '共313' in cb['desktop']['count_paragraph'] and '合計339' in cb['mobile']['count_paragraph']
out['count_fix_current_inspection']={'current_location':'README.md:15','current_quote':countline,'chapter_ids_count':len(chapterids),'index_count':len(index),'notebook_path_count':len(nbpaths),'all_three_id_path_sets_match':True,'chapter_counts':chaptercounts,'extra_ids':extras,'extra_count':extra,'current_total':358,'english_count_scope':{'path':'README_en.md','sha256':sha(R/'README_en.md'),'quote':engline},'original_fix_raw_receipts':[{'path':rpath(countfix),'sha256':sha(countfix),'raw_count':cf['count'],'diff':cf['diff']},{'path':rpath(countbrowser),'sha256':sha(countbrowser),'scope':cb['scope'],'original_desktop_count_quote':cb['desktop']['count_paragraph'],'original_mobile_count_quote':cb['mobile']['count_paragraph']}],'judgment_basis':'The old necessary fix was to keep README inventory accurate, not freeze313forever. Current332+26=358 and currentsource/index/notebooksets match, so original stale312/338defect has not returned. No current browser rebuild/execution was done.'}
mathline=next(x for x in readme.splitlines() if '公式由網站隨附的 MathJax 在瀏覽器排版。' in x);assert '公式由 MathJax 連網排版' not in readme
manifest=validate_manifest(j(R/'course/web/mathjax-assets.json'));assert manifest['version']=='3.2.2' and len(manifest['files'])==94
cfg=tomllib.loads((R/'zensical.toml').read_text());scripts=cfg['project']['extra_javascript'];assert 'javascripts/mathjax.js' in scripts and 'javascripts/mathjax/tex-chtml.js' in scripts and all(not x.startswith(('http://','https://')) for x in scripts if 'mathjax' in x)
js=(R/'course/web/mathjax.js').read_text();assert 'window.MathJax =' in js and 'window.MathJax.typesetPromise()' in js
export=(R/'scripts/export_course.py').read_text();assert 'mathjax = prepare_assets(ROOT)' in export and 'stage_assets(docs / "javascripts/mathjax", mathjax)' in export
stage=(R/'scripts/stage_mathjax.py').read_text();tree=ast.parse(stage);functions={x.name:ast.get_source_segment(stage,x) for x in tree.body if isinstance(x,ast.FunctionDef)};assert 'read_asset(source, item)' in functions['stage_assets'] and 'target.write_bytes(raw)' in functions['stage_assets'] and 'verify(raw, item)' in functions['read_asset']
oldfix=R/'docs/technical-reviews/artifacts/review-20261019-readme/check-current-01.json';assert sha(oldfix)=='10cd9be94cabb6dc9cb7c3add1387f06f1c292a911dfd4c4b4471fb6f50ca362';mf=j(oldfix);assert mf['single_replacement_exact'];oldnetwork=mf['network'];oldpage=next(x for x in oldnetwork if x['id']=='public_page');assert './javascripts/mathjax/tex-chtml.js' in oldpage['scripts']
main=next(x for x in manifest['files'] if x['path']=='tex-chtml.js');font=next(x for x in manifest['files'] if x['path']=='output/chtml/fonts/woff-v2/MathJax_Main-Regular.woff');assert main==mf['mathjax']['main'] and font==mf['mathjax']['font']
mathdeps=[]
for name in ['scripts/stage_mathjax.py','scripts/export_course.py','course/web/mathjax.js','course/web/mathjax-assets.json','zensical.toml']:mathdeps.append({'path':name,'sha256':sha(R/name)})
out['mathjax_fix_current_inspection']={'current_location':'README.md:72','current_sentence':'公式由網站隨附的 MathJax 在瀏覽器排版。','current_paragraph':mathline,'actual_current_dependencies':mathdeps,'read_current_functions':['stage_mathjax.validate_manifest/prepare_assets/read_asset/stage_assets','export_course.build prepare/stage calls','course/web/mathjax.js window.MathJax/typesetPromise','zensical.project.extra_javascript'],'manifest_validation_executed_without_download':True,'manifest_version':manifest['version'],'manifest_revision':manifest['revision'],'manifest_files':len(manifest['files']),'main_asset':main,'font_asset':font,'current_relative_page_script_configuration':[x for x in scripts if 'mathjax' in x],'current_browser_JS':js,'original_fix_raw_receipt':{'path':rpath(oldfix),'sha256':sha(oldfix),'original_actor':'/root/rv1019_technical_readme','original_acquisition_date':'2026-10-20','network_receipts':oldnetwork,'source_diff':mf['source_diff'],'single_replacement_exact':mf['single_replacement_exact']},'judgment_basis':'Read currentliteralclaim and actualstage/copy/localpath/browsertypesetcode; same94pinned3.2.2assets as originalfix receipt. Currentexport/config wholehashes changed across laterbook revisions, but the exactissue-related prepare/stage and relativecliententry chain remains. Priorcurl establishes originalfix provenance; current judgment comes from issue-specific currentbytes, not oldpass/still_fixed.','limits':'No newMathJaxasset acquisition, fullwebsite/browser/offlineacceptance ornetworkexecution. Building may fetch/cache assets; readersloadsitefiles; typesetting is browserJS, not a remote typesettingservice.'}
assert sha(R/full_refs['original']['path'])==full_refs['original']['sha256'] and sha(R/full_refs['current']['path'])==full_refs['current']['sha256']
out['completed_at']=datetime.now(timezone.utc).isoformat();p=A/'historical-inspection-v1.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'path':rpath(p),'sha256':sha(p),'bytes':p.stat().st_size,'started_at':out['started_at'],'completed_at':out['completed_at'],'packet_rows':[len(x['rows']) for x in packets],'current_count':[332,26,358],'mathjax_manifest_valid':True,'full_reports_unchanged':True},ensure_ascii=False))
