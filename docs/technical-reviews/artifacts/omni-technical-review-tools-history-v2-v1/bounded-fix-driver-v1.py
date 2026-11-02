"""Recheck exactly four owner issues; no full native approval or stage synthesis."""
import copy
import hashlib
import importlib.util
import json
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
TASK='/root/omni_technical_coordinator/technical_review_tools_history_v2'
ART=Path(__file__).parent
REPORT=ROOT/'docs/course-revision-20261029-omni/reviews/technical/technical_review_tools_history_v2/code-data-recheck-v3.json'
WORK=Path('/workspace/work/technical-review-tools-history-v2-bounded-fix-v1')
WORK.mkdir(exist_ok=False)
OUT=ART/'bounded-fix-execution-v1.json'
sys.path.insert(0,str(ROOT))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

started=datetime.now(UTC).isoformat()
assert sha(REPORT)=='633374c0a1c0184a14dac11c882b17239bad4907f7b3606288f63408cf95f068'
prior=json.loads(REPORT.read_bytes())
paths=[
 'docs/review-tools/incremental_revision.py',
 'docs/review-tools/historical_fix_rechecks_v2.py',
 'scripts/omni_revision/review_setup.py',
 'tests/test_historical_rechecks_v2.py',
 'docs/course-review-20261019/tools/issue_inventory.py',
 'docs/course-revision-20261029-omni/reviews/assembly-technical-intake-v2/actual-technical-collection-snapshot.json',
 'docs/course-revision-20261029-omni/reviews/assembly-technical-intake-v3/actual-technical-collection-snapshot.json',
 'docs/course-revision-20261029-omni/reviews/assembly-technical-intake-v3/dispositions-proposal.json',
]
prior_hash={x['path']:x['sha256'] for x in prior['checked_files']}
checked=[ref(ROOT/p) for p in paths]
assert all(x['sha256']==prior_hash[x['path']] for x in checked)
script=json.loads((ART/'execution-v1.json').read_bytes())['mutation_reproduction']['script']
ns={};exec(script.split('probes = [',1)[0],ns);ns['WORK']=WORK
code=[]
for name,mutation in [('valid_current_fixture',None),('continuity_current_created_before_original','chronology'),('technical_original_created_after_current','original_technical_time'),('technical_original_header_shell_wrong_source','original_technical_shell'),('continuity_original_unbound_source','original_continuity_source')]:
    target=WORK/name;target.mkdir();built=ns['fixture'].fixture(target)
    if mutation:ns[mutation](target,built)
    ns['sync_seals'](target,built)
    try:ns['fixture'].check(target,built);error=None
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    assert bool(error)==bool(mutation),(name,error)
    code.append({'name':name,'expected':'reject' if mutation else 'accept','observed':'reject' if error else 'accept','error':error,'scope':'Synthetic fixture native entry only; not genuine stage intake.'})
api=load('bounded_fix_api',ROOT/'docs/review-tools/incremental_revision.py')
setup=load('bounded_fix_setup',ROOT/'scripts/omni_revision/review_setup.py')
inventory=load('bounded_fix_inventory',ROOT/'docs/course-review-20261019/tools/issue_inventory.py')
packets=[]
for number,expected in [(1,'88b792f0765f944e2cfd9c19f3eec710302929b3ce70fb72343a35fa3d6bf57c'),(2,'1598b3a29cd28985496dcd07cd97ce7fb27de63da4b6f75ee92a3f4d29f2a907'),(3,'2b0d8faefbe15c8eccec152232954ad9feeb63625b5a31a96b8e1855510251de')]:
    path=ROOT/f'docs/course-revision-20261029-omni/reviews/assembly-technical-intake-v{number}/packet.json'
    assert sha(path)==expected;packets.append(json.loads(path.read_bytes()))
normalize=lambda xs:Counter(json.dumps(x,ensure_ascii=False,sort_keys=True) for x in xs)
collections=[api.document(ROOT,p['actual_technical_collection']) for p in packets]
join=[]
for number in (2,3):
    packet=packets[number-1];collection=collections[number-1]
    raw=api.document(ROOT,packet['actual_raw_ledger'])
    assert normalize(collection['collection_index']+collection['historical_intake_events'])==normalize(raw['collection_index'])
    assert not(normalize(collections[0]['collection_index'])-normalize(raw['collection_index']))
    assert len(collection['historical_intake_events'])==1
    assert len({(x['path'],x['sha256']) for x in collection['collection_index']})==len(collection['collection_index'])
    m=api.document(ROOT,packet['manifest']);tasks={x['task']:x for x in api.document(ROOT,packet['actual_assignments'])['tasks']}
    proof=next(x['technical'] for x in m['pages'] if x['page_id']=='19.24');report=api.document(ROOT,proof['report'])
    window=proof.get('current_dispatch_window',tasks[api.owner(report)]['dispatch_window'])
    matches=[x for x in collection['collection_index'] if (x['path'],x['sha256'])==(proof['report']['path'],proof['report']['sha256'])]
    assert len(matches)==1 and matches[0]['received_at']==proof['current_collection']['received_at']=='2026-10-29T13:39:49.818265+00:00'
    assert api.at(proof['original_collection']['received_at'])<=api.at(matches[0]['received_at'])
    def native_receipt(frame,claim):
        try:setup.check_receipt(api,{'omni-mainline-01':{'technical':frame}},proof['report'],report,claim,window)
        except Exception as exc:return type(exc).__name__+': '+str(exc)
        return 'accepted'
    result=native_receipt(collection,proof['current_collection']);assert result=="KeyError: 'completed_at'",result
    mixed=copy.deepcopy(collection);mixed['collection_index']+=copy.deepcopy(collection['historical_intake_events'])
    reinsert=native_receipt(mixed,proof['current_collection']);assert '缺唯一collection_index' in reinsert
    wrong=native_receipt(collection,{**proof['current_collection'],'received_at':'2026-10-29T13:37:06.304004+00:00'});assert '缺唯一collection_index' in wrong
    join.append({'packet_version':number,'collection':packet['actual_technical_collection'],'raw':packet['actual_raw_ledger'],'raw_count':len(raw['collection_index']),'formal_count':len(collection['collection_index']),'history_count':1,'partition_exact_and_old_events_preserved':True,'unique_formal_event_received_at':matches[0]['received_at'],'complete_native_receipt_result':result,'classification':'Unique join repaired. Genuine stage remains open; no complete receipt/stage pass.','negative_reinsert_interim':reinsert,'negative_claim_interim_time':wrong})
etrain=[]
for number in (2,3):
    packet=packets[number-1];m=api.document(ROOT,packet['manifest']);live={x['page_id']:x for x in api.document(ROOT,m['source_freeze'])['pages']}
    task_roles={x['task']:x['role'] for x in api.document(ROOT,packet['actual_assignments'])['tasks']}
    proposal=api.document(ROOT,packet['current_disposition_proposal'])
    for i in (309,310):
        item=proposal['items'][i];assert item['issue_id']=='E-TRAIN-02'
        errors=inventory.check_decision(ROOT,m,item,live,task_roles)
        expected=['fix requires actual verification references','fix does not cover affected pages/files'] if number==2 else []
        assert errors==expected,(number,i,errors)
        negatives=[]
        if number==3:
            for mode in ('verification','scope'):
                mutant=copy.deepcopy(item)
                if mode=='verification':mutant['verification']=[]
                else:mutant['affected_pages']=[];mutant['affected_files']=[]
                rejected=inventory.check_decision(ROOT,m,mutant,live,task_roles)
                expected_bad=['fix requires actual verification references','fix does not cover affected pages/files'] if mode=='verification' else ['fix does not cover affected pages/files']
                assert rejected==expected_bad,(mode,rejected)
                negatives.append({'removed':mode,'errors':rejected})
        etrain.append({'packet_version':number,'proposal':packet['current_disposition_proposal'],'json_pointer':'/items/'+str(i),'errors':errors,'negative_mutations':negatives})
assert all(sha(ROOT/x['path'])==x['sha256'] for x in checked)
assert sha(REPORT)=='633374c0a1c0184a14dac11c882b17239bad4907f7b3606288f63408cf95f068'
result={'schema_version':1,'kind':'four_issue_bounded_fix_reexecution_not_whole_tool_audit','reviewer_task':TASK,'started_at':started,'created_at':datetime.now(UTC).isoformat(),'command':'uv run --extra cpu python '+Path(__file__).relative_to(ROOT).as_posix(),'environment':{'python':sys.version,'platform':platform.platform(),'device':'CPU/offline; no model training'},'original_bounded_revise_report':ref(REPORT),'checked_files':checked,'checked_count':len(checked),'code_five_cases':code,'receipt_join_v2_v3':join,'E_TRAIN_02_v2_v3':etrain,'source_hashes_unchanged_before_after':True,'actual_complete_stage_or_manifest_approval':False,'author_reason_adoption_used_as_verdict':False}
with OUT.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps({'execution':ref(OUT),'checked_files':len(checked),'code_cases':len(code),'join_versions':len(join),'E_TRAIN_entries':len(etrain)},ensure_ascii=False))
