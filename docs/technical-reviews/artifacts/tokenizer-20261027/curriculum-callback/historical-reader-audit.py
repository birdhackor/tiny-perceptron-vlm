"""Post-original historical fix and fresh-reader source/range audit only."""
from pathlib import Path
import json,hashlib,datetime,importlib.util,subprocess,sys,re
R=Path.cwd();A=Path(__file__).parent;D=Path('docs/course-review-20261027-tokenizer/reviews/reader-curriculum-callback')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':digest(p)}
def load(p):return json.loads(Path(p).read_bytes())
def deref(r):assert digest(r['path'])==r['sha256'];return load(r['path'])
x={'checked_at':datetime.datetime.now(datetime.UTC).isoformat(),'command':'uv run --no-sync python docs/technical-reviews/artifacts/tokenizer-20261027/curriculum-callback/historical-reader-audit.py','environment':{'python':sys.version.split()[0],'device':'cpu'},'scope':'Only after own independent technical-original-01 saved: own recheck of old issue current wording/link, plus structural and content-scope check of fresh-reader evidence. Not a claim that technical reviewer performed the reader first-read.'}
original=Path('docs/course-review-20261027-tokenizer/reviews/technical-curriculum-callback/technical-original-01.json');x['independent_original']=ref(original);assert load(original)['recorded_at']<x['checked_at']
p=Path('docs/course-review-20261019/dispositions-continuity-01.json');item=load(p)['items'][71];old=deref(item['report']);issue=old['pages'][1]['issues'][0];v=deref(item['verification'][0]['report'])['pages'][0]
body=Path('docs/curriculum.md').read_text();line=next(s for s in body.splitlines() if 'BPE（位元組對編碼' in s);chapter=Path('course/chapters/06.md').read_text();section=re.search(r'^## 6\.2 .*?(?=^## |\Z)',chapter,re.M|re.S).group(0)
assert '[6.2](../course/chapters/06.md#6.2)' in line and 'Byte Pair Encoding，BPE' in section
x['historical']={'disposition_ref':{**ref(p),'json_pointer':'/items/71'},'original_report':item['report'],'json_pointer':item['json_pointer'],'issue_id':item['issue_id'],'original_necessity':issue['necessary'],'original_quote':issue['quote'],'original_suggestion':issue['suggestion'],'historical_fix_verification':item['verification'],'current_quote':line,'target_heading':section.splitlines()[0],'target_name_sentence':next(s for s in section.splitlines() if 'Byte Pair Encoding' in s),'current_source_sha256':digest('docs/curriculum.md'),'current_figure_sha256':{'course/figures/curriculum_learning_flow.svg':digest('course/figures/curriculum_learning_flow.svg')},'result':'still_fixed','scope':'Only BPE Chinese name and direct readback target; no claim that reader has learned BPE mechanism.'}
ro=load(D/'reader-original-01.json');rc=load(D/'reader-current-01.json');cl=load(D/'reader-closure-01.json');pg=ro['pages'][0];op=deref(pg['session_evidence']['opening']);state=load(Path(pg['session_evidence']['opening']['path']).parent/'state.json')
assert op['reviewer_task']=='/root/reader_curriculum_callback_20261027'
raw=deref(ro['reading_plan']);assert raw['main_pages']==['curriculum'] and raw['context_pages']==[]
spec=importlib.util.spec_from_file_location('curriculum_units',R/'docs/course-revision-20261016/tools/reader.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
chunks=m.units(Path(ro['reading_scope']['source']['path']).read_text());notes=[];last=op['started_at'];nonadequate={}
for i,(unit,nref) in enumerate(zip(chunks,pg['session_evidence']['notes'],strict=True)):
 n=deref(nref);assert n['unit_index']==i and n['reviewer_task']==op['reviewer_task'];assert n['unit_sha256']==hashlib.sha256(unit.encode()).hexdigest();assert n['saved_at']>=last;last=n['saved_at']
 for field in ['understanding','materials_and_labels','expected_change','confusion_and_quote','missing_visuals']:assert n[field].strip()
 for j,z in enumerate(n['concept_checks']):
  for field in ['concept','identity','mechanism','need','usage','example_scope','source_quote','verdict']:assert str(z[field]).strip()
  if z['verdict']!='已足夠':nonadequate[(nref['path'],j)]=z['verdict']
 notes.append({'unit_index':i,'unit_sha256':n['unit_sha256'],'note_ref':nref,'saved_at':n['saved_at'],'understanding':n['understanding'],'concept_ids':[z['concept'].split('：')[0] for z in n['concept_checks']]})
assert len(notes)==16 and state['next_index']==16
assert digest('docs/curriculum.md')==ro['reading_scope']['source']['sha256']==op['source_sha256']
closures=cl['pages'][0]['checkpoint_resolutions'];seen=set();counts={}
for z in closures:
 n=deref(z['note']);key=(z['note']['path'],z['concept_index']);assert key in nonadequate and key not in seen;seen.add(key)
 assert z['original_verdict']==nonadequate[key] and n['concept_checks'][z['concept_index']]['verdict']==z['original_verdict'];assert z['reason'] and z['evidence']
 assert z['owner']==op['reviewer_task'];counts[z['final_status']]=counts.get(z['final_status'],0)+1
assert seen==set(nonadequate)
tok=next(z for z in closures if z['checkpoint_id']=='C02-tokenizer');assert tok['final_status']=='deferred' and tok['necessary'] is False
checks=[]
for name in ['reader-original-01.json','reader-current-01.json']:
 cmd=['uv','run','--no-sync','python','docs/course-review-20261019/tools/check_reader_original.py',str(D/name)];p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0;checks.append({'command':' '.join(cmd),'exit_code':p.returncode,'stdout':p.stdout})
visual=load(D/'visual-receipt-01.json');screfs=[]
for r in visual['screenshots']:assert digest(r['path'])==r['sha256'];screfs.append({'sha256':r['sha256'],'viewport':r['viewport']})
x['fresh_reader']={'reports':[ref(D/n) for n in ['reader-original-01.json','reader-current-01.json','reader-closure-01.json','visual-receipt-01.json']],'opening':pg['session_evidence']['opening'],'opening_source_sha256':op['source_sha256'],'source_scope':ro['reading_scope'],'state_next_index':state['next_index'],'main_units':len(chunks),'all_unit_hashes_match':True,'notes_in_order':notes,'concept_checks_total':sum(len(deref(r)['concept_checks']) for r in pg['session_evidence']['notes']),'nonadequate_checks':len(nonadequate),'closure_counts':counts,'all_nonadequate_closure_refs_exact':True,'tokenizer_closure':tok,'format_checks_run_by_technical_reviewer':checks,'screenshot_hashes_checked':screfs,'finding':'Complete source/range evidence matches freeze-03. Reader explicitly leaves BPE operation unknown and defers it; only route/name recognition is present. Other method deferrals likewise do not prove learned mechanisms. No reader pass used as substitute for own independent technical reading.'}
x['result']='Historical BPE route/name still fixed; current fresh-reader 16-unit scope/notes and all20 closure refs match. Own original unchanged.'
(A/'historical-reader-audit.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'result':x['result'],'units':len(notes),'closures':counts,'original_sha256':digest(original)},ensure_ascii=False))
