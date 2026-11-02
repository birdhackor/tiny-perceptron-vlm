"""Audit immutable packet identities and actual data; never synthesize a seal."""
import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).with_name('intake-data-execution-v3-v5.json')
TASK='/root/omni_technical_coordinator/technical_review_tools_history_v2'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

api=load('intake_callback_native',ROOT/'docs/review-tools/incremental_revision.py')
setup=load('intake_callback_setup',ROOT/'scripts/omni_revision/review_setup.py')
validator=load('intake_callback_validator',ROOT/'scripts/check_technical_reviews.py')
inventory=load('intake_callback_inventory',ROOT/'docs/course-review-20261019/tools/issue_inventory.py')
packet_dir=ROOT/'docs/course-revision-20261029-omni/reviews/assembly-technical-intake-v3'
packet_path=packet_dir/'packet.json'
assert sha(packet_path)=='2b0d8faefbe15c8eccec152232954ad9feeb63625b5a31a96b8e1855510251de'
packet=json.loads(packet_path.read_bytes())
for value in packet.values():
    if isinstance(value,dict) and 'path' in value and 'sha256' in value:api.artifact(ROOT,value)
m=api.document(ROOT,packet['manifest']);assignments=api.document(ROOT,packet['actual_assignments'])
collection=api.document(ROOT,packet['actual_technical_collection']);raw=api.document(ROOT,packet['actual_raw_ledger'])
old_packet_path=ROOT/'docs/course-revision-20261029-omni/reviews/assembly-technical-intake-v1/packet.json';old_packet=api.document(ROOT,{'path':old_packet_path.relative_to(ROOT).as_posix(),'sha256':'88b792f0765f944e2cfd9c19f3eec710302929b3ce70fb72343a35fa3d6bf57c'})
old_m=api.document(ROOT,old_packet['manifest']);old_collection=api.document(ROOT,old_packet['actual_technical_collection'])
tasks={x['task']:x for x in assignments['tasks']}
assert len(tasks)==len(assignments['tasks'])
assert all(x['task'] not in m['author_tasks'] and x['context']=='fresh' for x in assignments['tasks'])
readers={t for t,x in tasks.items() if x['role']=='reader'}
task_roles={t:x['role'] for t,x in tasks.items()}
freeze=api.document(ROOT,m['source_freeze']);pages={p['page_id']:p for p in freeze['pages']}
_,owners=api.technical_reports(m,ROOT)
capture=api.at(m['intake_snapshot_created_at']);start=api.at(collection['started_at'])

def receipt(proof,key,report,current):
    reference=proof[key]
    matches=[x for x in collection['collection_index'] if (x['path'],x['sha256'])==(reference['path'],reference['sha256'])]
    assert len(matches)==1
    event=matches[0];task=api.owner(report)
    window=proof.get('current_dispatch_window',tasks[task]['dispatch_window']) if current else tasks[task]['dispatch_window']
    assert start<=api.at(window['not_before'])<=api.at(report['created_at'])<=api.at(event['received_at'])<=capture
    assert event['reviewer_task']==task and event['lesson_id']==report['lesson_id']
    assert reference['sha256'] in collection['report_sha256s']
    return event['received_at']

current_rows=[]
for row in m['pages']:
    proof=row['technical']
    if proof.get('wave')!='omni-mainline-01' or proof.get('mode')!='review':continue
    pid=row['page_id'];page=pages[pid];current=api.document(ROOT,proof['report']);original=api.document(ROOT,proof['original_report'])
    body=api.artifact(ROOT,{'path':page['snapshot'],'sha256':page['source_sha256']}).decode()
    errors=validator._validate(ROOT,ROOT/page['source'],pid,body,current,owners,readers)
    assert not errors,(pid,errors)
    setup.check_original(original,proof['original_source_freeze'],pid,pid,api.owner(current),validator,api,
                         original_source_snapshots=proof.get('original_source_snapshots',[]))
    original_at=receipt(proof,'original_report',original,False);current_at=receipt(proof,'report',current,True)
    assert api.at(original_at)<=api.at(current_at) and api.at(original['created_at'])<=api.at(current['created_at'])
    current_rows.append({'page_id':pid,'task':api.owner(current),'current':proof['report'],'original':proof['original_report'],
                         'schema_source_original_valid':True,'open_stage_receipts_unique_and_ordered':True,
                         'original_snapshots':len(proof.get('original_source_snapshots',[]))})
assert len(current_rows)==59
bundle=api.document(ROOT,packet['numbered_bundle_at_capture']);aux=api.document(ROOT,bundle['section_freeze'])
auxpages={p['page_id']:p for p in aux['pages']};allowners=copy.deepcopy(owners)
for entry in bundle['sections']:
    data=api.document(ROOT,entry['report']);allowners.setdefault(api.owner(data),[]).append(entry['lesson_id'])
sections=[]
for entry in bundle['sections']:
    sid=entry['lesson_id'];page=auxpages[sid];current=api.document(ROOT,entry['report']);original=api.document(ROOT,entry['original_report'])
    body=api.artifact(ROOT,{'path':page['snapshot'],'sha256':page['source_sha256']}).decode()
    errors=validator._validate(ROOT,ROOT/page['source'],sid,body,current,allowners,readers);assert not errors,(sid,errors)
    setup.check_original(original,entry['original_section_freeze'],sid,page['canonical_page_id'],api.owner(current),validator,api,
                         original_source_snapshots=entry.get('original_source_snapshots',[]))
    at0=receipt(entry,'original_report',original,False);at1=receipt(entry,'report',current,True)
    assert api.at(at0)<=api.at(at1)
    original_issues=set(api.reported_issues(original));closures=current.get('original_issue_resolutions',[])
    assert len(closures)==len(original_issues) and {x['json_pointer'] for x in closures}==original_issues
    sections.append({'lesson_id':sid,'current':entry['report'],'original':entry['original_report'],
                     'schema_source_original_receipts_valid':True,'original_issues_preserved':len(original_issues)})
assert len(sections)==26

normalize=lambda values:Counter(json.dumps(x,sort_keys=True,ensure_ascii=False) for x in values)
assert not(normalize(old_collection['collection_index'])-normalize(raw['collection_index']))
assert normalize(collection['collection_index']+collection['historical_intake_events'])==normalize(raw['collection_index'])
assert len(raw['collection_index'])==len(collection['collection_index'])+1 and len(collection['historical_intake_events'])==1
assert len({(x['path'],x['sha256']) for x in collection['collection_index']})==len(collection['collection_index'])
proof=next(x['technical'] for x in m['pages'] if x['page_id']=='19.24');report=api.document(ROOT,proof['report'])
window=proof.get('current_dispatch_window',tasks[api.owner(report)]['dispatch_window'])
def native_receipt(frame,claimed):
    try:setup.check_receipt(api,{'omni-mainline-01':{'technical':frame}},proof['report'],report,claimed,window)
    except Exception as exc:return type(exc).__name__+': '+str(exc)
    return 'accepted'
old_error=native_receipt(old_collection,proof['current_collection']);new_error=native_receipt(collection,proof['current_collection'])
assert '缺唯一collection_index' in old_error and new_error=="KeyError: 'completed_at'"
injected=copy.deepcopy(collection);injected['collection_index']+=copy.deepcopy(collection['historical_intake_events'])
injected_error=native_receipt(injected,proof['current_collection']);assert '缺唯一collection_index' in injected_error
wrong_claim={**proof['current_collection'],'received_at':'2026-10-29T13:37:06.304004+00:00'}
wrong_error=native_receipt(collection,wrong_claim);assert '缺唯一collection_index' in wrong_error
join={'old_error':old_error,'new_error':new_error,'new_error_classification':'unique formal join passes; genuine stage is still open, so completed_at is absent. No complete check_receipt pass claimed.',
      'negative_reinsert_interim':injected_error,'negative_claim_interim_time':wrong_error,
      'formal_current_received_at':proof['current_collection']['received_at'],
      'formal_original_received_at':proof['original_collection']['received_at'],
      'old_raw_events_preserved':True,'new_raw_partition_exact':True,'raw_count':len(raw['collection_index']),'formal_count':len(collection['collection_index']),'historical_count':1}

catalog=api.document(ROOT,packet['author_semantic_catalog_not_final_dispositions']);author_reason_checks=[]
for item in catalog['items']:
    anchor=item['original_anchor'];data=api.document(ROOT,anchor['report']);issue=api.node(data,anchor['json_pointer'])
    assert isinstance(issue,dict)
    assert all(isinstance(item.get(k),str) and item[k].strip() for k in ('source_support','learning_benefit','change_burden','reason'))
    assert item['final_necessity'] in ('necessary','optional') and item['actual_final_verification_pending'] is True
    api.artifact(ROOT,item['reason_source'])
    author_reason_checks.append({'anchor':anchor,'necessity':item['final_necessity'],'reason_present_and_pointer_real':True,
                                 'not_independent_semantic_approval':True})
proposal=api.document(ROOT,packet['current_disposition_proposal']);old_dispositions=api.document(ROOT,old_m['dispositions'])
assert len(proposal['items'])==341 and len(old_dispositions['items'])==283
assert proposal['items'][:257]==old_dispositions['items'][:257]
assert len({(x['report']['sha256'],x['json_pointer']) for x in proposal['items']})==341
per_item=[]
for i,item in enumerate(proposal['items']):
    if item.get('issue_id')=='E-TRAIN-02':
        errors=inventory.check_decision(ROOT,m,item,pages,task_roles)
        assert errors==[]
        per_item.append({'json_pointer':'/items/'+str(i),'errors':errors})
etrain_negatives=[]
for i,item in enumerate(proposal['items']):
    if item.get('issue_id')!='E-TRAIN-02':continue
    no_ver=copy.deepcopy(item);no_ver['verification']=[]
    no_ver_errors=inventory.check_decision(ROOT,m,no_ver,pages,task_roles)
    assert no_ver_errors==['fix requires actual verification references','fix does not cover affected pages/files']
    no_scope=copy.deepcopy(item);no_scope['affected_pages']=[];no_scope['affected_files']=[]
    no_scope_errors=inventory.check_decision(ROOT,m,no_scope,pages,task_roles)
    assert no_scope_errors==['fix does not cover affected pages/files']
    etrain_negatives.append({'json_pointer':'/items/'+str(i),'remove_verification':no_ver_errors,'remove_scope':no_scope_errors})
global_gate=api.audit_manifest(ROOT,ROOT/packet['manifest']['path'],'all');assert global_gate['status']=='failed'
try:numbered_result=setup.check_numbered(m,intake=True);numbered_error=None
except Exception as exc:numbered_error=type(exc).__name__+': '+str(exc)
assert numbered_error is None and numbered_result['status']=='schema_checked_collection_pending' and numbered_result['sections']==26
try:api.check_dispositions(ROOT,m,('reader','technical','continuity'),pages,task_roles,{})
except Exception as exc:dispositions_error=type(exc).__name__+': '+str(exc)
assert dispositions_error and 'unresolved author dispositions' in dispositions_error

from scripts import export_course
kernel=api.document(ROOT,packet['current_kernel_bundle_receipt']);index=json.loads((ROOT/'course/lesson-index.json').read_bytes())
index_by_id={x['id']:x for x in index};kernel_checked=0
for entry in kernel['records']:
    notebook=ROOT/index_by_id[entry['lesson']]['notebook'];executed=Path(entry['bundle_path'])
    assert sha(notebook)==entry['notebook_sha256'] and sha(executed)==entry['executed_sha256']
    export_course.checked_notebook(json.loads(notebook.read_bytes()),executed)
    kernel_checked+=1
assert kernel_checked==331

ui_ref=api.document(ROOT,packet['previous'])['author_ui_optional_adoption'];ui=api.document(ROOT,ui_ref);old=api.document(ROOT,ui['raw_reports'][0]);new=api.document(ROOT,ui['raw_reports'][1])
assert old['reviewer_task']==new['reviewer_task']=='/root/omni_reader_entries'
assert new['original_auxiliary_report']['sha256']==ui['raw_reports'][0]['sha256']
assert new['issue_resolution']['id']=='UI-COLAB-01' and new['issue_resolution']['final_status']=='resolved'
assert new['issue_resolution']['original_necessary'] is False and new['carryforward']['canonical_sessions_created']==0
evidence=api.document(ROOT,new['evidence_receipt']);assert sha(ROOT/evidence['implementation']['path'])==evidence['implementation']['sha256']
for previous in evidence['prior_auxiliary_artifacts_verified_unchanged']:api.artifact(ROOT,previous)
for source in evidence['source_pins']:assert sha(ROOT/source['path'])==source['sha256']
viewed_paths=[evidence['previous_official_ui_evidence_carried']['landing'],
              evidence['previous_official_ui_evidence_carried']['notebook_file_menu'],
              evidence['personally_viewed_png_receipts'][1]]
for entry in viewed_paths:assert sha(Path(entry['path']))==entry['sha256']
ui_checks={'adoption':ui_ref,'same_auxiliary_owner':True,'unchanged_optional_necessity':True,
           'current_implementation_sha256':evidence['implementation']['sha256'],'no_new_canonical_reader_session':True,
           'screenshots_personally_viewed_by_this_reviewer_in_tool_before_this_receipt':viewed_paths,
           'inspection':'Stored official landing visibly has Upload notebook, stored notebook File menu has Upload notebook, fix02 mobile footer distinguishes both and shows limits. No live reopening, login, upload or cloud execution.'}

source_scope=api.document(ROOT,packet['tool_input_scope']);assert len(source_scope['checked_file_scope'])==210
assert all(sha(ROOT/x['path'])==x['sha256'] for x in source_scope['checked_file_scope'])
result={'schema_version':1,'kind':'actual_bounded_packet_data_recheck_not_phase_acceptance','reviewer_task':TASK,
        'created_at':datetime.now(UTC).isoformat(),'command':'uv run --extra cpu python '+Path(__file__).relative_to(ROOT).as_posix(),
        'environment':{'python':sys.version,'platform':sys.platform,'device':'CPU/read-only source and metadata'},
        'packet_v1':ref(old_packet_path),'packet_v3':ref(packet_path),'packet_v2':packet['previous'],'formal_collection':packet['actual_technical_collection'],
        'current_count':59,'canonical_current_schema_original_and_open_receipts':current_rows,'numbered_count':26,'numbered_schema_original_and_open_receipts':sections,
        'join_issue_recheck':join,'author_reason_checks':author_reason_checks,'proposal':{'reference':packet['current_disposition_proposal'],
        'item_count':341,'accepted_257_dicts_exact':True,'declared_pending_count':proposal['pending_count'],'empty_fixed_verifications':sum(x.get('action')=='fixed' and not x.get('verification') for x in proposal['items']),
        'E_TRAIN_02_errors':per_item,'E_TRAIN_02_negative_mutations':etrain_negatives,'formal_dispositions_collection':m['dispositions_collection']},
        'actual_guards':{'native_all_status':global_gate['status'],'native_all_failures':global_gate['failures'],
                        'numbered_intake_error':numbered_error,'dispositions_error':dispositions_error},
        'kernel_bundle_checked_without_execution':kernel_checked,'auxiliary_ui_chain':ui_checks,
        'stage_status':collection['status'],'completed_at_present':'completed_at' in collection,'known_pending':packet['known_pending'],
        'limits':'59/100 integrated; source14 bundle4 has only passed intake schema, with all genuine stage/graph closure still pending; schema/source/receipt checks are not independent re-reading of lesson science. No new seal, no actual full gate pass, no GPU/training/publication.'}

previous_packet=api.document(ROOT,packet['previous'])
previous_raw=api.document(ROOT,previous_packet['actual_raw_ledger'])
assert not(normalize(previous_raw['collection_index'])-normalize(raw['collection_index']))
previous_formal=api.document(ROOT,previous_packet['actual_technical_collection'])
previous_manifest=api.document(ROOT,previous_packet['manifest'])
assert bundle['canonical_source_freeze']==m['source_freeze'] and bundle['section_freeze']=={'path':'docs/course-revision-20261029-omni/reviews/numbered-freeze-v12.json','sha256':'4290356672497263cd82abff1904eff118c8d8163aee3f985de541790bd5669c'}
issue_index=api.document(ROOT,bundle['issue_inventory'])
assert issue_index==setup.section_issue_index(bundle,issue_index['author_task'],api)
collectors=[]
for pid in ('course','first-steps','training','glossary'):
    proof_parent=next(x['technical'] for x in m['pages'] if x['page_id']==pid)
    report_parent=api.document(ROOT,proof_parent['report'])
    bindings=[x for x in report_parent['artifacts'] if x.get('kind')=='source_snapshot' and 'numbered-section-reviews' in x.get('path','')]
    binds_current=any(x.get('path')==m['numbered_section_reviews']['path'] and x.get('sha256')==m['numbered_section_reviews']['sha256'] for x in bindings)
    for binding in bindings:api.artifact(ROOT,binding)
    collectors.append({'page_id':pid,'report':proof_parent['report'],'binds_bundle4':binds_current,'actual_bindings':bindings,'classification':'pending genuine new whole collector; packet declares this gap'})
assert proposal['pending_count']==15 and len(proposal['pending_verification'])==15
assert sum(x.get('action')=='fixed' and not x.get('verification') for x in proposal['items'])==13
bad_supported=[i for i,x in enumerate(proposal['items']) if 'supported' in x.get('status','') and (not x.get('verification') or not(x.get('affected_pages') or x.get('affected_files')))]
assert not bad_supported
previous_proposal=api.document(ROOT,previous_packet['current_disposition_proposal'])
reader_changed=[{'index':i,'changed_keys':sorted(k for k in set(previous_proposal['items'][i])|set(proposal['items'][i]) if previous_proposal['items'][i].get(k)!=proposal['items'][i].get(k))} for i in range(257,283) if previous_proposal['items'][i]!=proposal['items'][i]]
result['packet_v3_specific_checks']={'previous_raw_events_preserved':True,'numbered_intake':numbered_result,'bundle4':m['numbered_section_reviews'],'index4':bundle['issue_inventory'],'collectors':collectors,'bad_supported':bad_supported,'reader_late_ref_changes':reader_changed,'formal_collection_v2':previous_packet['actual_technical_collection'],'formal_collection_v3':packet['actual_technical_collection'],'accepted257_exact':True}

with OUT.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':ref(OUT),'canonical':59,'numbered':26,'join':join,'guards':result['actual_guards'],
                  'catalog_reason_count':len(author_reason_checks),'kernel_checked_without_execution':kernel_checked},ensure_ascii=False),flush=True)
