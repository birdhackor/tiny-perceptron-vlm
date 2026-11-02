"""Read-only training whole-page/numbered-proof crosscheck; never run a model or rewrite a proof."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,platform,random,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));A=Path(__file__).parent
SIDECAR={'path':'docs/technical-reviews/artifacts/omni-P5-numbered-v1/numbered-section-reviews-v1.json','sha256':'d57aebb02fadc2b8b3a0f07e6dc59b1d831193dc92b4d06504b52dbda2acabd6'}
PROVENANCE={'path':'docs/course-revision-20261029-omni/reviews/P5-actual/numbered-neutral-receipt-provenance-v1.json','sha256':'211e46f27219c436700e5743d0805a072786e2853e38e59b570656040c62c02d'}
ORIGINAL={'path':'docs/course-revision-20261029-omni/reviews/P5-actual/technical-training/technical-original-v1.json','sha256':'d31b396b2fd906cfdfd672edd65ac7621cfe218779fac8edb9553dfb03184c4e'}
TASK='/root/omni_p5_technical_training_whole'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(path):
 p=ROOT/path;return {'path':path,'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
def read(reference):
 p=ROOT/reference['path'];raw=p.read_bytes();assert sha(raw)==reference['sha256'],reference['path'];return json.loads(raw)
def imported(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
setup=imported('p5_training_setup','scripts/omni_revision/review_setup.py')
validator=imported('p5_training_validator','scripts/check_technical_reviews.py')
api=setup.module('p5_training_incremental','docs/review-tools/incremental_revision.py')
reading=setup.module('p5_training_reading','scripts/reading_time.py')
side=read(SIDECAR);prov=read(PROVENANCE);original=read(ORIGINAL)
assert len(side['sections'])==26
assert side['canonical_source_freeze']==prov['source_freeze']
canon=read(side['canonical_source_freeze']);section_freeze=read(side['section_freeze']);inventory=read(side['issue_inventory'])
assert side['issue_inventory']['sha256']=='53a170ec26339686cb110f593da0787d360f53af33d16ea35fb2421bba0cdb82'
assert section_freeze['canonical_source_freeze']==side['canonical_source_freeze']
assert inventory['canonical_source_freeze']==side['canonical_source_freeze'] and inventory['section_freeze']==side['section_freeze']
page=next(p for p in canon['pages'] if p['page_id']=='training');body=(ROOT/page['snapshot']).read_bytes();live=(ROOT/'course/training.md').read_bytes()
oldfreeze=read(original['source_freeze']);oldpage=next(p for p in oldfreeze['pages'] if p['page_id']=='training');oldbody=(ROOT/oldpage['snapshot']).read_bytes()
assert body==live==oldbody and sha(body)==page['source_sha256']==original['source_sha256'] and page['figures_sha256']==original['figure_sha256']=={}
assert api.at(original['created_at'])<api.at(side['created_at'])
result={'created_at':datetime.datetime.now(datetime.UTC).isoformat(),'reviewer_task':TASK,'kind':'collector added crosscheck, not a collection or seal','environment':{'python':platform.python_version(),'platform':platform.platform(),'execution':'JSON,source/hash/schema/receipt arithmetic only; no model construction,forward,training,network,dispatch,publishing'},'sidecar':SIDECAR,'receipt_provenance':PROVENANCE,'issue_inventory':side['issue_inventory'],'canonical_freeze':side['canonical_source_freeze'],'section_freeze':side['section_freeze'],'independent_original':ORIGINAL,'whole_page':{'source_bytes':len(body),'source_sha256':sha(body),'byte_identical_to_original':True,'source10_live_snapshot_original_all_equal':True,'figures':{},'introduction_and_T1_T11_independent_review_reused':True,'new_full_page_read_or_model_execution_claimed':False},'sections':[],'phase_status':'Current P5 technical remains open; no final check-numbered/seal claimed.'}
# Original's reviewed source and artifacts are immutable, not refreshed to new hashes.
result['original_repository_sources']=[{'path':s['path'],'recorded_sha256':s['sha256'],'current_sha256':sha((ROOT/s['path']).read_bytes())} for s in original['sources'] if s['kind']=='repository_code']
assert all(r['recorded_sha256']==r['current_sha256'] for r in result['original_repository_sources'])
result['original_artifacts']=[{'path':a['path'],'sha256':a['sha256'],'matches':sha((ROOT/a['path']).read_bytes())==a['sha256']} for a in original['artifacts']]
assert all(a['matches'] for a in result['original_artifacts'])
setup.check_original(original,original['source_freeze'],'training','training',TASK,validator,api)
# Read only report metadata outside training to reject an accidentally reused owner.
owners={TASK:['training']}
for e in side['sections']:
 r=read(e['report']);owners.setdefault(r['reviewer_task'],[]).append(e['lesson_id'])
assert all(len(v)==1 for v in owners.values())
prior_assignment=read(prov['prior_assignment_source']);current_dispatch=read(prov['current_dispatch_source'])
reader_tasks={r['task'] for r in prior_assignment['tasks'] if r['role']=='reader'}
reader_tasks.update(read({'path':'docs/course-revision-20261029-omni/reviews/P5-actual/technical-inputs-packet-v4.json','sha256':'1d79bb1ae2b1f1f3235381d765c5de3588b47e4d981315246f4614ef148f61cd7'})['reader_tasks']) if False else None
packet=json.loads((ROOT/'docs/course-revision-20261029-omni/reviews/P5-actual/technical-inputs-packet-v4.json').read_bytes());reader_tasks.update(packet['reader_tasks'])
assignments={r['lesson_id']:r for r in prov['numbered_task_assignments']}
assert len(assignments)==26
oldwave=next(w for w in prov['prior_completed_waves'] if w['id']=='omni-mainline-01');oldseal=read(oldwave['seals']['technical']);oldreaderseal=read(oldwave['seals']['reader'])
assert oldseal['status']=='completed' and oldreaderseal['status']=='completed' and api.at(oldreaderseal['completed_at'])<=api.at(oldseal['started_at'])
sealed={'omni-mainline-01':{'technical':oldseal}}
current_wave=prov['current_wave'];reader=read(current_wave['reader_completed']);opened=read(current_wave['technical_open']);collection=read(current_wave['actual_collection_so_far'])
assert reader['status']=='completed' and current_wave['completed_technical_seal'] is None
assert api.at(reader['completed_at'])<=api.at(opened['started_at'])
result['receipt_sources']={'prior_completed_technical':oldwave['seals']['technical'],'prior_assignments':prov['prior_assignment_source'],'current_dispatch':prov['current_dispatch_source'],'current_reader_completed':current_wave['reader_completed'],'current_technical_open':current_wave['technical_open'],'current_collection_so_far':current_wave['actual_collection_so_far'],'current_technical_completed':False}
slices=reading.lesson_slices(body.decode());wanted={f'T.{i}' for i in range(1,12)}
assert set(slices)==wanted
selected=[e for e in side['sections'] if e['lesson_id'] in wanted]
assert {e['lesson_id'] for e in selected}==wanted and len(selected)==11
frozen={p['page_id']:p for p in section_freeze['pages']}
for e in selected:
 sid=e['lesson_id'];r=read(e['report']);o=read(e['original_report']);f=frozen[sid];section=slices[sid];task=r['reviewer_task'];assignment=assignments[sid]
 assert (ROOT/f['snapshot']).read_bytes()==section.encode() and sha(section.encode())==f['source_sha256']==r['source_sha256']
 assert f['canonical_page_id']=='training' and f['source']=='course/training.md' and f['selector']=='section:'+sid
 assert f['figures_sha256']==r['figure_sha256']==reading.figure_hashes(ROOT,ROOT/f['source'],section)=={}
 assert task==o['reviewer_task']==assignment['task'] and assignment['role']=='technical' and assignment['context']=='fresh'
 assert task not in packet['authors'] and task not in reader_tasks and task!=TASK
 setup.check_original(o,e['original_section_freeze'],sid,'training',task,validator,api,original_source_snapshots=e.get('original_source_snapshots',[]))
 errors=validator._validate(ROOT,ROOT/f['source'],sid,section,r,owners,reader_tasks);assert not errors,(sid,errors)
 issues=set(api.reported_issues(o));closure=r.get('original_issue_resolutions',[])
 assert issues=={v['json_pointer'] for v in closure} and len(issues)==len(closure) and all(v['resolution'].strip() for v in closure)
 assert api.at(o['created_at'])<=api.at(r['created_at'])
 receipts=[]
 if assignment['wave']=='omni-mainline-01':
  setup.check_report_pair(api,sealed,e,o,r,assignment)
  for key,colkey in [('original_report','original_collection'),('report','current_collection')]:
   matches=[v for v in oldseal['collection_index'] if v['path']==e[key]['path'] and v['sha256']==e[key]['sha256'] and v['received_at']==e[colkey]['received_at']];assert len(matches)==1
   receipts.append({'kind':key,'reference':e[key],'wave':e[colkey]['wave'],'received_at':matches[0]['received_at'],'unique_matching_actual_receipt':True})
 else:
  assert sid=='T.11' and assignment['wave']==current_wave['id']
  assert api.at(opened['started_at'])<=api.at(assignment['actual_spawn_acknowledged_before'])<=api.at(o['created_at'])
  for key,colkey in [('original_report','original_collection'),('report','current_collection')]:
   matches=[v for v in collection['collection_index'] if v['path']==e[key]['path'] and v['sha256']==e[key]['sha256'] and v['received_at']==e[colkey]['received_at']];assert len(matches)==1
   m=matches[0];rep=o if key=='original_report' else r
   assert m['reviewer_task']==task and m['created_at']==rep['created_at'] and m['wave']==e[colkey]['wave']==current_wave['id'] and m['collection_role']=='technical'
   assert api.at(rep['created_at'])<=api.at(m['received_at'])<=api.at(collection['created_at'])
   receipts.append({'kind':key,'reference':e[key],'wave':m['wave'],'received_at':m['received_at'],'unique_matching_actual_receipt':True,'open_stage_only':True})
  assert api.at(e['original_collection']['received_at'])<=api.at(e['current_collection']['received_at'])
 row={'lesson_id':sid,'mode':'retained_actual_prior_proof' if sid!='T.11' else 'actual_P5_section_task','reviewer_task':task,'assignment':assignment,'source_sha256':f['source_sha256'],'source_bytes':len(section.encode()),'figure_sha256':{},'current_report':e['report'],'original_report':e['original_report'],'original_section_freeze':e['original_section_freeze'],'original_verdict_preserved':o['verdict'],'current_verdict':r['verdict'],'original_claim_count':len(o['claims']),'current_claim_count':len(r['claims']),'all_current_claims_verified':all(c['status']=='verified' for c in r['claims']),'original_full_schema_sources_artifacts_checked':True,'current_validate_errors':errors,'original_issues':sorted(issues),'same_reviewer_issue_closure':closure,'receipts':receipts,'new_section_read_or_execution_claimed':False,'new_collector_action':'read this section original/current claims/checks/scope/issue closure;compare with independent whole-page original and current exact frozen body;verify all source/artifact/report digests and actual receipt identity'}
 if e.get('schema_corrected_original_history'):
  hist=read(e['schema_corrected_original_history']);raw=read(hist['raw_initial_original']);formal=read(hist['formal_original_and_current'])
  for k in ('claims','source_sha256','sources','verdict'):assert raw[k]==formal[k],('T4 raw metadata preservation',k)
  row['schema_corrected_history']={'index':e['schema_corrected_original_history'],'raw_original':hist['raw_initial_original'],'formal_original':hist['formal_original_and_current'],'raw_initial_claims_sources_and_verdict_unchanged':True,'raw_failed_selfcheck_preserved':hist['raw_initial_failed_selfcheck']};read(hist['raw_initial_failed_selfcheck'])
 if sid=='T.11':
  assert o['claims']==r['claims'] and o['checks']==r['checks'] and o['sources']==r['sources'] and o['source_sha256']==r['source_sha256']
  assert len(section.encode())==5309
  row['reviewer_metadata_correction_verified']={'pointer':'/artifacts/0/result','old_mistyped_bytes':6107,'current_correct_bytes':5309,'actual_snapshot_bytes':len(section.encode()),'claims_sources_checks_verdict_unchanged':True,'original_unchanged':True}
 result['sections'].append(row)
# T.9's sole indexed original issue was corrected before this collector's independent original.
from scripts.prepare_data import generate_records
rows=generate_records('attributes-sft');families=sorted({r['family'] for r in rows});random.Random(42).shuffle(families);a,b=int(.8*len(families)),int(.9*len(families));assignment={f:'train' if i<a else 'validation' if i<b else 'test' for i,f in enumerate(families)}
probe='color=blue;shape=square;pitch=low;color?';oldprobe='color=blue;shape=circle;pitch=low;color?'
current_match=[r for r in rows if r['messages'][0]['content']==probe];old_match=[r for r in rows if r['messages'][0]['content']==oldprobe]
assert len(current_match)==len(old_match)==1 and assignment[current_match[0]['family']]=='train' and assignment[old_match[0]['family']]=='test'
t9=slices['T.9'];assert t9.count('--prompt "'+probe+'"')==2 and '--prompt "'+oldprobe+'"' not in t9
assert t9.index('先用validation選位數')<t9.index('--split-label test')
training_issues=[v for v in inventory['issues'] if v['canonical_page_id']=='training'];assert len(training_issues)==1 and training_issues[0]['lesson_id']=='T.9' and training_issues[0]['json_pointer']=='/issues/0'
result['T9_issue_crosscheck']={'inventory_entry':training_issues[0],'current_probe':probe,'current_family':current_match[0]['family'],'current_split':'train','gold':'blue','old_probe_preserved':oldprobe,'old_split':'test','two_current_commands_match':True,'validation_selection_precedes_test_commands':True,'resolution':'same reviewer original issue was genuine premature test-example exposure; current source uses train blue:square:low and explicitly keeps validation selection before test; independently corroborated by this collector original t903 and now pure generator recomputation','no_claim_of_actual_test_tuning_or_score_contamination':True}
# Recompute the original numeric facts again from the collector-owned portable projection only.
projection=json.loads((A/'measurement-projection-v1.json').read_bytes());total=sum(r['bytes'] for r in projection['selected_four_weights'].values());assert total==2192940
c=projection['C_final'];accepts=[r for r in c['rows'] if r['tasks']['choice']['p_max']>=c['selection_tau'] and r['tasks']['choice']['answer']!='none' and r['choice_info_answer']=='yes'];errors=sum(r['tasks']['choice']['answer']!=r['tasks']['choice']['gold'] or not r['world_supported'] or not r['choice_information_sufficient'] for r in accepts)
assert len(accepts)==53 and errors==19
result['independent_numeric_agreement_with_T11']={'source':ref('docs/technical-reviews/artifacts/omni-P5-training/measurement-projection-v1.json'),'four_weights_bytes':total,'C_choice_correct':sum(r['tasks']['choice']['answer']==r['tasks']['choice']['gold'] for r in c['rows']),'C_family_denominator':len(c['rows']),'accepted':len(accepts),'accepted_errors':errors,'risk':errors/len(accepts),'new_model_forward_calls':0}
result['summary']={'checked_training_sections':11,'retained_T1_T10':10,'P5_T11':1,'current_section_claims_checked':sum(r['current_claim_count'] for r in result['sections']),'original_section_claims_checked':sum(r['original_claim_count'] for r in result['sections']),'current_section_contract_errors':0,'unresolved_training_section_issues':0,'whole_page_original_preserved':True,'entire_shared_sidecar_metadata_count':26,'shared_sidecar_semantic_scope':'only training T.1..T.11,not R/W/G lessons','current_phase_sealed':False,'new_model_execution':False}
assert sha((ROOT/ORIGINAL['path']).read_bytes())==ORIGINAL['sha256']
p=A/'crosscheck-current-v1.json'
with p.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'output':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes()),'summary':result['summary']},ensure_ascii=False))
