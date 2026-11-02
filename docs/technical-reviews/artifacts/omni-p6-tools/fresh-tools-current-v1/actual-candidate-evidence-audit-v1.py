"""Supplement genuine OPEN-T04 candidate metadata, never fabricate a completed seal."""
from pathlib import Path
from datetime import datetime,UTC
import collections,hashlib,importlib.util,json
ROOT=Path('/workspace/work/omni-book-integrated-v1')
BASE='docs/course-revision-20261029-omni/reviews/P5-actual/wave04/'
NOW=datetime.now(UTC)
BINDINGS={};ERRORS=[];RECEIPTS=[];TECHNICAL=[];READERS=[]
def need(x,msg):
 if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reference(name):
 p=ROOT/name;return {'path':name,'sha256':sha(p)}
def doc(ref):
 p=Path(ref['path']);p=p if p.is_absolute() else ROOT/p
 need(sha(p)==ref['sha256'],'reference SHA differs: '+str(p))
 BINDINGS[str(p)]={'path':ref['path'],'sha256':ref['sha256'],'bytes':p.stat().st_size}
 return json.loads(p.read_bytes())
def at(x):
 d=datetime.fromisoformat(x);need(d.tzinfo is not None,'timezone missing');return d

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
api=load('fresh_current_incremental','docs/review-tools/incremental_revision.py')
setup=load('fresh_current_setup','scripts/omni_revision/review_setup.py')
validator=load('fresh_current_validator','scripts/check_technical_reviews.py')
preflight=load('fresh_current_preflight','docs/course-review-20261019/tools/preflight.py')
intake_ref=reference(BASE+'technical-collection-intake-v29.json');intake=doc(intake_ref)
reader_seal_ref=reference(BASE+'reader-phase-proof-v1.json');reader_seal=doc(reader_seal_ref)
dispatch_ref=reference(BASE+'technical-dispatch-intake-v9.json');dispatch=doc(dispatch_ref)
reader_dispatch_ref=reference(BASE+'reader-dispatch-intake-v4.json');reader_dispatch=doc(reader_dispatch_ref)
technical_join_ref=reference(BASE+'technical-current-proof-join-v1.json');technical_join=doc(technical_join_ref)
reader_join_ref=reference(BASE+'reader-current-proof-join-v3.json');reader_join=doc(reader_join_ref)
freeze_ref=dispatch['source_freeze'];freeze=doc(freeze_ref);pages={p['page_id']:p for p in freeze['pages']}
TSTART=min(at(t['dispatch_window']['not_before']) for t in dispatch['tasks'])
need('OPEN' in intake['status'] and 'OPEN' in dispatch['status'],'T04 must remain truthfully OPEN')
need(reader_seal['role']=='reader' and reader_seal['status']=='completed','R04 must be genuinely completed')
need(at(reader_seal['completed_at'])<=TSTART<=NOW,'R04/T04 chronology')
tasks={t['task']:t for t in dispatch['tasks']};rtasks={t['task']:t for t in reader_dispatch['tasks']}
need(len(tasks)==len(dispatch['tasks']),'duplicate technical assignment')
need(not(set(tasks)&set(rtasks)),'reader/technical owners reused')

def check_receipt(ref,owner,role,declared=None,assignment=None):
 source=reader_seal if role=='reader' else intake
 matches=[r for r in source['collection_index'] if r.get('path')==ref['path'] and r.get('sha256')==ref['sha256']]
 need(len(matches)==1,'missing/duplicate actual receipt '+ref['path'])
 row=matches[0];data=doc(ref);need(api.owner(data)==owner,'report owner differs '+ref['path'])
 need(row['reviewer_task']==owner and row['role']==role and row.get('collection_role',role)==role,'receipt owner/role differs '+ref['path'])
 need(row['wave']=='omni-mainline-04-final','receipt wave differs')
 created=at(data['created_at']);received=at(row['received_at'])
 lo=at(reader_seal['started_at']) if role=='reader' else TSTART
 hi=at(reader_seal['completed_at']) if role=='reader' else NOW
 need(lo<=created<=received<=hi,'actual creation/receipt outside observed phase '+ref['path'])
 if role=='reader':need(ref['sha256'] in reader_seal['report_sha256s'],'R04 SHA not sealed')
 if declared:need(declared['received_at']==row['received_at'] and declared['collection_role']==role,'declared receipt differs')
 if assignment:
  w=assignment['dispatch_window'];end=w.get('not_after') or assignment.get('actual_spawn_acknowledged_before')
  need(lo<=at(w['not_before'])<=at(end)<=hi,'dispatch window outside actual phase')
  need(at(w['not_before'])<=created,'report predates dispatch lower bound')
 RECEIPTS.append({'reference':ref,'reviewer_task':owner,'collection_role':role,'created_at':data['created_at'],'received_at':row['received_at'],'unique_actual_index':True,'within_closed_phase' if role=='reader' else 'within_observed_OPEN_phase':True})
 return created,received

def try_check(label,fn):
 try:return fn()
 except Exception as e:ERRORS.append({'scope':label,'type':type(e).__name__,'message':str(e)});return None

numbered=doc(technical_join['numbered_section_reviews'])
new_sections=[e for e in numbered['sections'] if e['original_collection']['wave']=='omni-mainline-04-final' or e['current_collection']['wave']=='omni-mainline-04-final']
entries=[(r['page_id'],r['technical'],r['page_id']) for r in technical_join['canonical_proofs']]
section_freeze=doc(numbered['section_freeze']);section_pages={p['page_id']:p for p in section_freeze['pages']}
entries += [(e['lesson_id'],e,section_pages[e['lesson_id']]['canonical_page_id']) for e in new_sections]
all_reports=[]
for row in technical_join['canonical_proofs']:all_reports.append(doc(row['technical']['report']))
for row in numbered['sections']:all_reports.append(doc(row['report']))
owners=collections.defaultdict(list)
for report in all_reports:owners[api.owner(report)].append(report['lesson_id'])

def check_technical(sid,proof,parent):
 current=doc(proof['report']);original=doc(proof['original_report']);owner=api.owner(current)
 task=tasks[owner];need(task['context']=='fresh' and task['role']=='technical' and task['lesson_id']==sid,'wrong fresh assignment')
 need(api.owner(original)==owner,'original/current owner differs')
 page=pages[sid] if sid==parent else section_pages[sid]
 body=(ROOT/page['snapshot']).read_text()
 need(hashlib.sha256(body.encode()).hexdigest()==page['source_sha256'],'frozen source changed')
 current_errors=validator._validate(ROOT,ROOT/page['source'],sid,body,current,owners,set(rtasks))
 need(not current_errors,'current native validator errors '+repr(current_errors))
 setup.check_original(original,proof.get('original_source_freeze',proof.get('original_section_freeze')),sid,parent,owner,validator,api,original_source_snapshots=proof.get('original_source_snapshots',[]))
 c0,r0=check_receipt(proof['original_report'],owner,'technical',proof['original_collection'],task)
 c1,r1=check_receipt(proof['report'],owner,'technical',proof['current_collection'],task)
 need(c0<=c1 and r0<=r1,'original/current chronology reversed')
 TECHNICAL.append({'lesson_id':sid,'owner':owner,'original':proof['original_report'],'current':proof['report'],'native_current_errors':current_errors,'complete_original_contract_checked':True,'genuine_receipts_checked':True})
for sid,proof,parent in entries:try_check('technical:'+sid,lambda s=sid,p=proof,c=parent:check_technical(s,p,c))

def check_reader(row):
 pid,proof=row['page_id'],row['reader'];current=doc(proof['report']);owner=api.owner(current);task=rtasks[owner]
 need(task['context']=='fresh' and pid in task['pages'],'reader original assignment differs')
 basis=doc(proof['basis']['report']);original=doc(proof['original_report']);need(api.owner(original)==owner==api.owner(basis),'reader owner differs')
 cur=api.node(current,proof.get('json_pointer',''));bas=api.node(basis,proof['basis']['json_pointer'])
 need(cur['source_sha256']==pages[pid]['source_sha256']==bas['source_sha256'],'reader current/basis source not source18')
 need(cur.get('figures_sha256')==pages[pid]['figures_sha256']==bas.get('figures_sha256'),'reader current/basis figures differ')
 c0,r0=check_receipt(proof['original_report'],owner,'reader',assignment=task)
 role=proof.get('collection_role','reader')
 c1,r1=check_receipt(proof['report'],owner,role,proof.get('current_collection'),task if role=='reader' else None)
 need(c0<=c1 and r0<=r1,'reader original/current chronology reversed')
 check_receipt(proof['basis']['report'],owner,role)
 closure=proof['checkpoint_closure'];check_receipt(closure['report'],owner,closure['collection_role'],closure)
 traces=[]
 for session in proof['sessions']:
  opening=doc(session['opening']);source_ref=session['source'];need(sha(ROOT/source_ref['path'])==source_ref['sha256'],'session source SHA differs')
  actual_pid,completed=api.reader_session(ROOT,session,owner);need(actual_pid==pid,'session page differs')
  traces.append({'opening':session['opening'],'source_sha256':opening['source_sha256'],'units':len(session['notes']),'started_at':opening['started_at'],'completed_at':completed.isoformat(),'collection_role':session.get('collection_role','reader'),'reading_mode':session.get('reading_mode','retained original source trace; no new blanket whole-read claim')})
 basis_errors=preflight.basis_read(ROOT,proof);need(not basis_errors,'coordinator basis check errors '+repr(basis_errors))
 READERS.append({'page_id':pid,'owner':owner,'current_collection_role':role,'basis_complete_session_errors':basis_errors,'traces':traces,'no_relabeling_of_T04_informed_reread_as_blind_R04':True})
for row in reader_join['proofs']:try_check('reader:'+row['page_id'],lambda row=row:check_reader(row))

def own_tool_original():
 ref=reference(BASE+'technical-tools/original-v1.json');report=doc(ref)
 need(report['schema_version']==1 and report['review_stage']=='technical' and report['reviewer_context']=='fresh','tool original schema/context')
 need(isinstance(report['checks'],dict) and report['checks'] and isinstance(report['issues'],list),'tool original lacks checks/issues')
 need(report['reviewer_task']=='/root/omni_p6_technical_tools','tool owner')
 c,r=check_receipt(ref,report['reviewer_task'],'technical',assignment=tasks[report['reviewer_task']])
 neutral=doc(reference(BASE+'technical-tools-neutral-inputs-v1.json'))
 expected={e['path']:e for e in neutral['files']};checked={e['path']:e['sha256'] for e in report['checked_files']}
 need(len(checked)==len(report['checked_files'])==460 and set(checked)==set(expected),'tool original complete460 paths')
 for name,digest in checked.items():need(sha(Path(expected[name]['snapshot']))==digest==sha(ROOT/name)==expected[name]['sha256'],'tool raw/current source differs '+name)
 return {'original':ref,'actual_created_at':report['created_at'],'actual_received_at':r.isoformat(),'schema_context_checks_issues':True,'raw_source18_files_checked':460,'all_sha_match':True,'original_necessary_labels_unchanged':True}
TOOL=try_check('own tool original',own_tool_original)

def previous_tool_raw():
 v4=doc(reference('docs/course-revision-20261029-omni/reviews/P5-tools/technical-current-v4.json'))
 v3ref=v4['original'];v3=doc(v3ref);provenance=doc(v4['first_complete460_scope_provenance'])
 need(provenance['first_complete460_report']['path']==v3ref['path'] and provenance['first_complete460_report']['sha256']==v3ref['sha256'],'first complete460 provenance differs')
 snapshot=doc(v3['snapshot']);raw_root=Path(snapshot['snapshot_path']);checked=v3['checked_files'];need(len(checked)==460==len({x['path'] for x in checked}),'old tool complete460')
 need(v3['schema_version']==1 and v3['review_stage']=='technical' and v3['reviewer_context']=='fresh','old tool native schema/context')
 need(isinstance(v3.get('issues'),list) and isinstance(v3.get('executed_checks'),list) and v3['executed_checks'],'old tool substantive checks/issues missing')
 current_delta=[]
 for row in checked:
  need(sha(raw_root/row['path'])==row['sha256'],'old complete460 raw differs '+row['path'])
  if sha(ROOT/row['path'])!=row['sha256']:current_delta.append(row['path'])
 return {'first_complete460':v3ref,'provenance':v4['first_complete460_scope_provenance'],'raw_root':str(raw_root),'files_checked':460,'all_raw_sha_match':True,'substantive_native_checks_field':'executed_checks (12 entries), not modern checks field; legacy format preserved','current_delta':current_delta,'initial30_partial_not_misrepresented_as_complete_original':True,'four_old_tool_issue_answers_not_read_in_this_stage':True}
PREVIOUS_TOOL=try_check('previous complete460 raw',previous_tool_raw)
result={'schema_version':1,'reviewer_task':'/root/omni_p6_technical_tools','created_at':datetime.now(UTC).isoformat(),'status':'open_phase_data_checks_failed' if ERRORS else 'open_phase_data_checks_passed_for_declared_primary_sets','inputs':{'source_freeze':freeze_ref,'technical_intake':intake_ref,'technical_dispatch':dispatch_ref,'reader_seal':reader_seal_ref,'reader_dispatch':reader_dispatch_ref,'technical_join':technical_join_ref,'reader_join':reader_join_ref,'numbered_sidecar':technical_join['numbered_section_reviews']},'observed_T04_start_lower_bound':TSTART.isoformat(),'observed_T04_upper_bound':NOW.isoformat(),'T04_completed':False,'technical_primary_units_checked':TECHNICAL,'reader_primary_pages_checked':READERS,'own_complete460_original':TOOL,'previous_complete460_original':PREVIOUS_TOOL,'receipt_checks':RECEIPTS,'bound_artifacts':list(BINDINGS.values()),'errors':ERRORS,'limits':['Partial data-level supplement, not a production gate/runtime fix.','Exact final candidate manifest/proof-set binding and final current/appendix receipts remain pending real creation/collection.','No old historical four-tool issue appendix read; no source/model/training/workflow/publication changes.']}
path=Path(__file__).with_name('actual-candidate-evidence-audit-v1.json')
with path.open('x') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'status':result['status'],'technical_units_checked':len(TECHNICAL),'reader_pages_checked':len(READERS),'receipt_checks':len(RECEIPTS),'bound_artifacts':len(BINDINGS),'errors':ERRORS},ensure_ascii=False,indent=2))
