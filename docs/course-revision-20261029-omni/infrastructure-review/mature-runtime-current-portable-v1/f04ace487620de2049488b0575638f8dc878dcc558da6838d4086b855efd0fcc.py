"""真 current finish/finance；source admission另測，這裡只替換source context及遠端GET/writer。"""

import copy, hashlib, json, os, socket, subprocess, sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT=Path('/workspace/work/omni-models-v1');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
import torch
from scripts.omni_revision import research_execution as r, research_recovery as recovery
from scripts.omni_revision.finance import digest, reservation_identity, validate_state

def read(p): return json.loads(Path(p).read_bytes())
ledger=json.loads(read(HERE.parent/'original/provider-ledger-response.json')['structuredContent']['content'])
freeze=read(HERE/'source-snapshot-manifest.json'); source_files={x['path']:x['sha256'] for x in freeze['files']}
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in source_files.items())
sources={'revision':'ee799d24fd1ea26108af709ec41e49db63d58e31','files':source_files,'audit_sha256':'f'*64}
foreign=copy.deepcopy(next(x for x in ledger['attempts'] if x['attempt_id']=='cpu-core-40988311100-1')['receipt'])
foreign['release_body']=read(HERE.parent/'mr07-original/provider-foreign-core-release.json')['body']
original_res=read('/workspace/work/omni-kev-first-preparation-failure-41004743884/reservation.json')['attempt']
attempt=original_res['attempt_id']; reserved=copy.deepcopy(original_res);purpose=json.loads(reserved['purpose'])
purpose.update(sources_sha256=digest(sources),controller_audit_sha256=sources['audit_sha256'])
reserved['purpose']=json.dumps(purpose,sort_keys=True);reserved['reservation_sha256']=digest(reservation_identity(reserved))
base=copy.deepcopy(ledger);target=next(x for x in base['attempts'] if x['attempt_id']==attempt);target.clear();target.update(copy.deepcopy(reserved));validate_state(base,datetime.now(UTC))
tag,body=r.release_identity(attempt,sources['revision'])
receipt={'schema':'omni-mature-controller-receipt-v2','attempt':attempt,'job':'kev','source_revision':sources['revision'],'controller_audit_sha256':sources['audit_sha256'],'release_tag':tag,'release_body':body,'input_payload_sha256':purpose['input_sha256'],'config_sha256':purpose['job_config_sha256'],'core_selection_sha256':purpose['core_selection_sha256'],'core_final_opening_sha256':purpose['core_final_opening_sha256'],'preparation_recovery':None,'success':True,'compute_stopped':True,'resources_verified':True,'cleanup_checks':{'private_stop_fixture':True},'release_id':428663042,'assets':[{'id':690960942,'name':'input.zip','bytes':434670,'sha256':purpose['input_sha256'],'roundtrip_verified':True},{'id':999000001,'name':'result.zip','bytes':17,'sha256':'1'*64,'roundtrip_verified':True}],'result_exported_before_stop':True,'research_final_opened':True}
env={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':'birdhackor/tiny-perceptron-vlm','GITHUB_REF':'refs/heads/main','GITHUB_EVENT_NAME':'workflow_dispatch','GITHUB_RUN_ID':'41004743884','GITHUB_RUN_ATTEMPT':'1','MODAL_SANDBOX_V2':'0','MODAL_ENVIRONMENT':'main','OMNI_ARTIFACT_OUTCOME':'success'}
events=[]; real_bound=r.bound
def no_external(*a,**k): raise AssertionError('unexpected network/subprocess/NN')
def run(name,rec=None,res=None,target_ok=True):
 out=HERE/'finish-private-cases'/name;out.mkdir(parents=True,exist_ok=False);localres=copy.deepcopy(res or reserved);localrec=copy.deepcopy(rec or receipt)
 (out/'reservation.json').write_text(json.dumps({'attempt':localres}));(out/'research-receipt.json').write_text(json.dumps(localrec))
 calls=[];writes=[];bounds=[];construct=[]
 def api(endpoint):
  calls.append(endpoint)
  if endpoint.startswith('actions/'):
   return {'total_count':1,'artifacts':[{'id':12954523057,'name':'omni-'+attempt,'size_in_bytes':517015,'created_at':'2026-10-30T04:48:04Z','expires_at':'2026-11-06T04:48:03Z'}]}
  assert endpoint=='releases/428663042'
  return {'draft':True,'tag_name':tag,'body':body,'target_commitish':sources['revision'] if target_ok else '0'*40,'assets':[{'id':a['id'],'name':a['name'],'size':a['bytes']} for a in localrec['assets']]}
 class Store:
  def __init__(self): construct.append(True)
  def read(self):
   state=copy.deepcopy(base)
   if res is not None and 'preparation_recovery' in name:
    row=next(x for x in state['attempts'] if x['attempt_id']==attempt);row.clear();row.update(copy.deepcopy(localres))
   return state,'private-blob'
  def write(self,state,blob):
   assert blob=='private-blob';validate_state(state,datetime.now(UTC));writes.append(state);return {'private':'no remote write'}
 def bound(p): bounds.append(True);return real_bound(p)
 error=None
 with patch.dict(os.environ,env),patch.object(r,'checked_sources',return_value=copy.deepcopy(sources)),patch.object(r,'api',side_effect=api),patch.object(r,'GitHubLedger',Store),patch.object(r,'bound',side_effect=bound),patch.object(socket.socket,'connect',new=no_external),patch.object(subprocess,'run',new=no_external):
  try:r.finish_run('kev',attempt,out)
  except (ValueError,FileNotFoundError) as e:error={'type':type(e).__name__,'message':str(e)}
 row=next(x for x in writes[0]['attempts'] if x['attempt_id']==attempt) if writes else None
 result={'case':name,'error':error,'provider_gets':len(calls),'bound_calls':len(bounds),'writer_constructions':len(construct),'recorded_ledger_writes':len(writes),'result_status':row['status'] if row else None,'real_ledger_writes':0}
 events.append(result);return result

guard=torch.nn.modules.module.register_module_forward_pre_hook(no_external)
try:
 f=run('genuine_foreign_core_receipt',foreign);assert f['error'] and f['provider_gets']==f['bound_calls']==f['writer_constructions']==0
 for field in ('schema','attempt','job','source_revision','controller_audit_sha256','release_tag','release_body','input_payload_sha256','config_sha256','core_selection_sha256','core_final_opening_sha256','preparation_recovery'):
  bad=copy.deepcopy(receipt);bad[field]='foreign';x=run('receipt_'+field,bad);assert x['error'] and x['provider_gets']==x['bound_calls']==x['recorded_ledger_writes']==0
 for field in ('job','sources_sha256','controller_audit_sha256','ownership','input_sha256','job_config_sha256','core_selection_sha256','core_final_opening_sha256'):
  bad=copy.deepcopy(reserved);p=json.loads(bad['purpose']);p[field]='foreign';bad['purpose']=json.dumps(p);x=run('purpose_'+field,res=bad);assert x['error'] and x['provider_gets']==x['bound_calls']==x['recorded_ledger_writes']==0
 for field in ('attempt_id','revision','status'):
  bad=copy.deepcopy(reserved);bad[field]='foreign';x=run('reservation_'+field,res=bad);assert x['error'] and x['provider_gets']==x['recorded_ledger_writes']==0
 x=run('matching_complete_fixture');assert x['error'] is None and x['result_status']=='completed'
 bad=copy.deepcopy(receipt);bad['success']=False;bad['assets']=bad['assets'][:1];x=run('matching_failed_fixture',bad);assert x['result_status']=='failed-stopped'
 x=run('provider_wrong_target',target_ok=False);assert x['result_status']=='unresolved'
 bad=copy.deepcopy(receipt);bad['result_exported_before_stop']=False;x=run('no_export_before_stop',bad);assert x['result_status']=='unresolved'
 bad=copy.deepcopy(receipt);bad['research_final_opened']=False;x=run('no_research_opening',bad);assert x['result_status']=='unresolved'
 private=HERE/'finish-private-permit-root';path=private/recovery.PERMIT_NAME;path.parent.mkdir(parents=True);path.write_text('private fixture only\n');proof=recovery.lineage(hashlib.sha256(path.read_bytes()).hexdigest());res=copy.deepcopy(reserved);p=json.loads(res['purpose']);p.update(proof);res['purpose']=json.dumps(p,sort_keys=True);res['reservation_sha256']=digest(reservation_identity(res));rec=copy.deepcopy(receipt);rec['preparation_recovery']=proof
 with patch.object(r,'ROOT',private):
  x=run('matching_preparation_recovery_fixture',rec,res);assert x['result_status']=='completed'
  for key in recovery.LINEAGE_KEYS:
   bad=copy.deepcopy(rec);bad['preparation_recovery'][key]='foreign';x=run('preparation_recovery_'+key,bad,res);assert x['error'] and x['provider_gets']==x['recorded_ledger_writes']==0
  bad=copy.deepcopy(res);p=json.loads(bad['purpose']);p.pop('original_row_sha256');bad['purpose']=json.dumps(p);x=run('partial_preparation_recovery_lineage',rec,bad);assert x['error'] and x['provider_gets']==x['recorded_ledger_writes']==0
finally:guard.remove()
result={'schema':'omni-mature-runtime-current-finish-independent-probe-v1','created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'cases':events,'source_context_note':'checked_sources is a private fixture for identity-transition probes while operative permit/audit/commit are pending; actual admission is separately checked and remains blocked. Genuine foreign CPU receipt flags are unchanged. Matching successful mature results are private protocol fixtures, not NN observations.','real_ledger_writes':0,'real_release_writes':0,'modal_rpc_calls':0,'nn_forward_calls':0}
(HERE/'current-finish-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'all_assertions_passed':True,'cases':len(events),'genuine_foreign_rejected_before_provider_bound_writer':True,'matching_completed':True,'wrong_target_unresolved':True,'real_ledger_writes':0}))
