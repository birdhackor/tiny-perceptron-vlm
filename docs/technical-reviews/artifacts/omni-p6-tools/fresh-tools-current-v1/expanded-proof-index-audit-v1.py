"""Bind the complete T04 data projection and all actual callback intake; no seal/gate mutation."""
from pathlib import Path
from datetime import datetime,UTC
import collections,hashlib,importlib.util,json
ROOT=Path('/workspace/work/omni-book-integrated-v1');NOW=datetime.now(UTC)
BASE='docs/course-revision-20261029-omni/reviews/P5-actual/wave04/'
BOUND={};ERRORS=[];LIMITED=[];PROOFS=[];CALLBACKS=[]
def need(x,s):
 if not x:raise ValueError(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(name):return {'path':name,'sha256':sha(ROOT/name)}
def doc(r):
 p=Path(r['path']);p=p if p.is_absolute() else ROOT/p;need(sha(p)==r['sha256'],'SHA mismatch '+str(p));BOUND[str(p)]={'path':r['path'],'sha256':r['sha256'],'bytes':p.stat().st_size};return json.loads(p.read_bytes())
def at(s):
 v=datetime.fromisoformat(s);need(v.tzinfo is not None,'timestamp lacks timezone');return v
def node(x,p=''):
 for k in p.split('/')[1:] if p else []:
  k=k.replace('~1','/').replace('~0','~');x=x[int(k)] if isinstance(x,list) else x[k]
 return x
def load(name,path):
 s=importlib.util.spec_from_file_location(name,ROOT/path);v=importlib.util.module_from_spec(s);s.loader.exec_module(v);return v
api=load('expanded_incremental','docs/review-tools/incremental_revision.py')
index_ref=ref(BASE+'technical-evidence-proof-index-v1.json');index=doc(index_ref)
need(index_ref['sha256']=='401077a841f76acb54d976dd2066609e71058e484ff40bcf2ff4b10c88152018','index identity')
base=doc(index['base_projection']['report']);freeze=doc(index['source_freeze']);live={p['page_id']:p for p in freeze['pages']}
assignments=[]
for comp in index['assignment_components']:assignments+=node(doc(comp['report']),comp['array_pointer'])
tasks={t['task']:t for t in assignments};need(len(tasks)==len(assignments)==652,'actual assignment union differs')
need(all(t.get('context')=='fresh' and t['role'] in {'reader','technical','continuity'} for t in assignments),'assignment role/context')
authors=set(base['author_tasks']);need(not(set(tasks)&authors),'author assigned independent reviewer')
pages={p['page_id']:dict(p) for p in base['pages']}
for override in index['R_T_overrides']:
 for p in node(doc(override['join']),override['array_pointer']):pages[p['page_id']][override['proof_field']]=p[override['proof_field']]
need(set(pages)==set(live) and len(pages)==369,'current canonical page set differs')
phases={};earlier=[]
for wave in base['review_waves']:
 phases[wave['id']]={}
 for role,r in wave['seals'].items():
  phase=doc(r);need(phase['status']=='completed' and phase['role']==role,'earlier phase not genuinely completed')
  need(at(phase['started_at'])<=at(phase['completed_at'])<=NOW,'earlier phase chronology')
  phases[wave['id']][role]=phase;earlier.append({'wave':wave['id'],'role':role,'seal':r,'started_at':phase['started_at'],'completed_at':phase['completed_at']})
phase_sources=[doc(r) for r in index['actual_phase_and_receipt_sources']]
rseal,topen,tintake,bounded=phase_sources
need(rseal['status']=='completed' and 'OPEN' in topen['status'] and 'OPEN' in tintake['status'],'R04/T04 state differs')
need(at(rseal['completed_at'])<=at(topen['started_at'])<=NOW,'R04 must precede open T04')
phases['omni-mainline-04-final']={'reader':rseal,'technical':{'started_at':topen['started_at'],'observed_at':NOW.isoformat(),'collection_index':tintake['collection_index'],'status':'OPEN','role':'technical'}}
patches=api.author_patches(ROOT,base,live)
versions={pid:api.page_versions(ROOT,p,p,patches) for pid,p in live.items()}

def observe(label,fn):
 try:fn()
 except Exception as e:ERRORS.append({'scope':label,'type':type(e).__name__,'message':str(e)})

def selected(pid,role,proof):
 if proof['mode']=='inherit':
  before=doc(base['baseline_freeze']);old={p['page_id']:p for p in before['pages']}[pid]
  need(all(live[pid].get(k)==old.get(k) for k in ['source','selector','source_sha256','figures_sha256']),'inherit source differs')
  PROOFS.append({'page_id':pid,'role':role,'mode':'inherit','scope':'unchanged authenticated original baseline, not a fresh reread'});return
 need(proof['mode']=='review','unexpected ordinary proof mode')
 report=doc(proof['report']);original=doc(proof['original_report']);owner=api.owner(report);task=tasks[owner]
 need(owner not in authors and task['role']==role and api.owner(original)==owner,'selected role/owner differs')
 need(task['wave']==proof['wave'],'assignment/proof wave differs')
 need(pid in task.get('pages',[]) if role=='reader' else task.get('lesson_id')==pid,'assigned page differs')
 current=node(report,proof.get('json_pointer',''));orig=node(original,proof.get('original_json_pointer',''))
 need(current.get('page_id',current.get('lesson_id'))==pid==orig.get('page_id',orig.get('lesson_id')),'page member differs')
 need(current.get('verdict') in {'pass','已足夠'},'current verdict not pass')
 used=api.patched_version(current.get('source_sha256'),live[pid]['source_sha256'],versions[pid]);need(used is not None,'selected current source stale')
 need(current.get('figures_sha256',current.get('figure_sha256'))==live[pid]['figures_sha256'],'selected figure identity stale')
 actual_role=proof.get('collection_role',role);phase=phases[proof['wave']][actual_role]
 need(proof['report']['sha256'] in phase.get('report_sha256s',[proof['report']['sha256']]),'current report not sealed')
 matches=[r for r in phase.get('collection_index',[]) if r.get('path')==proof['report']['path'] and r.get('sha256')==proof['report']['sha256']]
 # Current T04 new primary metadata has already received stricter original/current checks in the separate47-receipt audit.
 if phase['status']=='OPEN':need(len(matches)==1,'open-phase current lacks unique actual receipt')
 if len(matches)==1:
  got=at(matches[0]['received_at']);need(at(phase['started_at'])<=got<=at(phase.get('completed_at',phase.get('observed_at'))),'selected receipt outside actual phase')
  for k in ['created_at','confirmed_at']:
   if isinstance(report.get(k),str):need(at(report[k])<=got<=NOW,'selected report creation/receipt reversed')
 elif proof['wave']=='omni-mainline-04-final':need(False,'new selected current lacks unique receipt')
 else:LIMITED.append({'page_id':pid,'role':role,'wave':proof['wave'],'boundary':'unchanged earlier native seal authenticates SHA; no new per-path receipt requirement invented retroactively','report':proof['report']})
 PROOFS.append({'page_id':pid,'role':role,'mode':'review','owner':owner,'wave':proof['wave'],'collection_role':actual_role,'source18_or_legal_patch_match':True,'patches':list(used),'current':proof['report'],'original':proof['original_report'],'unique_path_SHA_receipt':len(matches)==1})
for pid,row in pages.items():
 for role in ['reader','technical']:observe(pid+'/'+role,lambda p=pid,r=role,q=row[role]:selected(p,r,q))

# Every real new intake document is checked, including author, bounded-owner and transport observations.
all_indices=[('R04',rseal['collection_index']),('T04',tintake['collection_index']),('R04-bounded',bounded['collection_index'])]
for layer,rows in all_indices:
 seen=set()
 for row in rows:
  pair=(row['path'],row['sha256']);need(pair not in seen,'duplicate exact receipt within '+layer);seen.add(pair)
  def callback(row=row,layer=layer):
   report=doc(row);received=at(row['received_at']);owner=api.owner(report) or report.get('author_task') or report.get('collector_task')
   phase=rseal if layer.startswith('R04') else topen
   lo=at(phase['started_at']);hi=at(phase['completed_at']) if layer.startswith('R04') else NOW
   need(lo<=received<=hi,'callback receipt outside real phase')
   created=report.get('created_at',report.get('at'))
   if isinstance(created,str):need(at(created)<=received<=NOW,'callback created after actual receipt')
   if owner is not None:need(owner==row['reviewer_task'] or report.get('reviewer_task')==row['reviewer_task'],'callback owner/recorder differs')
   kind=row.get('kind','')
   if owner in authors:classification='author metadata; not an independent technical/reader verdict'
   elif 'transport' in row['path'] or 'aborted' in row['path']:classification='transport/exposure observation; not a full primary review'
   elif 'bounded' in row['path'] or 'own-fix' in row['path']:classification='same original-owner bounded callback; not a new fresh full review'
   elif 'appendix' in row['path']:classification='separately collected appendix; not a substitute for full original/current'
   else:classification='actual received document; primary membership resolved separately'
   CALLBACKS.append({'layer':layer,'reference':{'path':row['path'],'sha256':row['sha256']},'declared_owner':row['reviewer_task'],'actual_owner_or_recorder':owner,'created_at':created,'received_at':row['received_at'],'classification':classification})
  observe('callback:'+row['path'],callback)

# Independently rehash genuine-preservation inputs; don't adopt another reviewer's conclusions by citation alone.
retention_ref=ref('docs/course-revision-20261029-omni/reviews/P5-actual/coherence-entries/retention-genuine-proof-verification-v1.json');retention=doc(retention_ref)
retained=[]
for r in retention['artifact_bindings']:
 p=ROOT/r['path'];actual=sha(p)
 if actual!=r['expected_sha256']:ERRORS.append({'scope':'retained genuine artifact','path':r['path'],'message':'actual SHA differs'})
 retained.append({'path':r['path'],'sha256':actual,'matches_prior_binding':actual==r['expected_sha256']})
# Bind author inputs without classifying their adjudications as independent discoveries.
author_inputs=[{'reference':r,'keys':list(doc(r))} for r in index['author_verification_inputs']]
result={'schema_version':1,'reviewer_task':'/root/omni_p6_technical_tools','created_at':datetime.now(UTC).isoformat(),'status':'data_projection_failed' if ERRORS else 'data_projection_checks_passed_for_exact_T04_index','index':index_ref,'source_freeze':index['source_freeze'],'assignment_count':len(tasks),'assignment_role_counts':dict(collections.Counter(t['role'] for t in tasks.values())),'selected_R_T_proof_count':len(PROOFS),'ordinary_selected_proofs':PROOFS,'earlier_completed_phase_refs':earlier,'current_T04_status':'genuinely OPEN, not sealed','actual_new_intake_documents_checked':CALLBACKS,'all_intake_rows':sum(len(x) for _,x in all_indices),'independently_rehashed_preservation_artifacts':retained,'author_inputs_bound_as_author_metadata':author_inputs,'legacy_receipt_format_limits':LIMITED,'bound_artifacts':list(BOUND.values()),'errors':ERRORS,'pending':['Own current is not yet created/received.','No C04 or final author dispositions supplied.','Final ordinary metadata/proof capsule must replace the temporary base projection and receive a genuine final delta check.','Original production counterexamples remain accepted; these are supplemental checks of the exact present input set.']}
p=Path(__file__).with_name('expanded-proof-index-audit-v1.json')
with p.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'status':result['status'],'selected_proofs':len(PROOFS),'intake_rows':result['all_intake_rows'],'checked_intake_rows':len(CALLBACKS),'retention_artifacts':len(retained),'legacy_format_limits':len(LIMITED),'error_count':len(ERRORS),'errors':ERRORS[:20]},ensure_ascii=False,indent=2))
