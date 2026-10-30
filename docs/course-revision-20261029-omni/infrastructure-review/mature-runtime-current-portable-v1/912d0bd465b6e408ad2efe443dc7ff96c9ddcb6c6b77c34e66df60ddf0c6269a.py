"""獨立真前置/原ZIP唯讀整合與private permit反例；未建立operativepermit，沒有Modal/NN。"""

import base64, copy, hashlib, json, sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT=Path('/workspace/work/omni-models-v1');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
import torch
from scripts.omni_revision import research_recovery as recovery, research_execution as runner, research_core_inputs as core_inputs, core_final
from scripts.omni_revision.finance import digest

def read(p):return json.loads(Path(p).read_bytes())
response=read(HERE/'provider-ledger.json');raw=base64.b64decode(response['content']);assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==response['sha'];state=json.loads(raw)
row=next(x for x in state['attempts'] if x['attempt_id']==recovery.IDENTITY['attempt'])
assert digest(row)==recovery.IDENTITY['row_sha256'] and digest(row['receipt'])==recovery.IDENTITY['receipt_sha256']
assert state['reserved_total_usd']=='6.60'
live_release=read(HERE/'provider-old-input-release.json')
assert not (ROOT/recovery.PERMIT_NAME).exists()
missing=False
try:recovery.eligibility(state,'kev')
except ValueError as error:missing='缺本人' in str(error)
assert missing
private=HERE/'recovery-private-root';private.mkdir();original_path=private/recovery.ORIGINAL_NAME;original_path.parent.mkdir(parents=True);original_path.write_bytes((ROOT/recovery.ORIGINAL_NAME).read_bytes())
permit_path=private/recovery.PERMIT_NAME
permit={'schema':'omni-mature-preparation-recovery-permit-v1','status':'pass_for_exact_preparation_recovery','reviewer_task':'/root/omni_development_execution_review','created_at':datetime.now(UTC).isoformat(),'identity':recovery.IDENTITY,'original_runtime_review':{'path':recovery.ORIGINAL_NAME,'sha256':recovery.ORIGINAL_SHA},'maximum_new_attempts':1,'new_bound_usd':'4.30','source_path_proof':{'sdk_version':'1.6.0','first_write_failed_before_any_fs_rpc':True,'sha_marker_not_sent_by_failed_path':True,'provider_historical_NN_telemetry_directly_observed':False},'private_fixture_not_operative':True}
permit_path.write_text(json.dumps(permit))
events=[];gets=[]
def eligibility_case(name,current_state=None,current_permit=None,release=None,should_pass=False,job='kev'):
 permit_path.write_text(json.dumps(current_permit or permit));gets.clear()
 def get(endpoint):gets.append(endpoint);assert endpoint=='releases/428663042';return copy.deepcopy(release if release is not None else live_release)
 error=None;value=None
 with patch.object(recovery,'ROOT',private),patch.object(recovery,'api',side_effect=get):
  try:value=recovery.eligibility(copy.deepcopy(current_state if current_state is not None else state),job)
  except (ValueError,KeyError) as e:error=type(e).__name__
 assert (error is None)==should_pass,(name,error,value)
 events.append({'case':name,'accepted':error is None,'error_type':error,'provider_get_count':len(gets),'private_only':True})
 return value

proof=eligibility_case('genuine_exact_predecessor_and_live_asset',should_pass=True);assert proof==recovery.lineage(hashlib.sha256(permit_path.read_bytes()).hexdigest())
for status in ('reserved','unresolved','completed','failed-stopped'):
 s=copy.deepcopy(state);s['attempts'].append({'attempt_id':'private-second','status':status,'purpose':json.dumps({'job':'mature-kev'})});eligibility_case('second_same_job_'+status,s)
for key,value in [('recovery_of',recovery.IDENTITY['attempt']),('preparation_recovery_permit_sha256',proof['preparation_recovery_permit_sha256'])]:
 s=copy.deepcopy(state);s['attempts'].append({'purpose':json.dumps({'job':'mature-qwen25',key:value})});eligibility_case('permit_already_consumed_'+key,s)
for field,value in [('status','unresolved'),('revision','0'*40),('reserved_usd','0'),('attempt_id','foreign')]:
 s=copy.deepcopy(state);next(x for x in s['attempts'] if x['attempt_id']==recovery.IDENTITY['attempt'])[field]=value;eligibility_case('predecessor_'+field,s)
for field,value in [('phase','worker_stdout_until_export_ready'),('error_type','TimeoutError'),('input_payload_uploaded',True),('result_exported_before_stop',True),('compute_stopped',False),('resources_verified',False),('remaining_owned_sandboxes',['sb-unknown']),('stop_errors',{'sb-unknown':'unknown'})]:
 s=copy.deepcopy(state);next(x for x in s['attempts'] if x['attempt_id']==recovery.IDENTITY['attempt'])['receipt'][field]=value;eligibility_case('receipt_'+field,s)
for field,value in [('status','author_proposal_not_authorized'),('reviewer_task','/root'),('maximum_new_attempts',True),('maximum_new_attempts',2),('new_bound_usd','0'),('created_at','2026-10-30T04:00:00+00:00'),('created_at',(datetime.now(UTC)+timedelta(days=1)).isoformat())]:
 p=copy.deepcopy(permit);p[field]=value;eligibility_case('permit_'+field+str(value)[:4],current_permit=p)
for field,value in [('draft',False),('target_commitish','0'*40),('tag_name','foreign'),('body','foreign'),('id',1)]:
 rel=copy.deepcopy(live_release);rel[field]=value;eligibility_case('provider_'+field,release=rel)
for field,value in [('id',1),('name','result.zip'),('size',1),('digest','sha256:'+'0'*64)]:
 rel=copy.deepcopy(live_release);rel['assets'][0][field]=value;eligibility_case('asset_'+field,release=rel)
rel=copy.deepcopy(live_release);rel['assets'].append({'name':'result.zip'});eligibility_case('extra_provider_asset',release=rel)
assert eligibility_case('ordinary_first_qwen_still_once_scope',should_pass=True,job='qwen25') is None
s=copy.deepcopy(state);s['attempts'].append({'purpose':json.dumps({'job':'mature-qwen25'}),'status':'unresolved'});eligibility_case('qwen_prior_unknown_rejected',s,job='qwen25')

nn=[]
def forbid(module,inputs):nn.append(type(module).__name__);raise AssertionError('NN forward forbidden')
guard=torch.nn.modules.module.register_module_forward_pre_hook(forbid)
try:
 # 使用原已安裝且逐SHA核過的成品，重用同一底層唯讀驗證；不呼叫會拒覆寫既有evidence的freshActions wrapper。
 metadata=core_inputs.check_metadata('ee799d24fd1ea26108af709ec41e49db63d58e31')
 tag,asset=core_inputs.completed_core(state)
 selected=core_final.checked_selection(ROOT/'outputs/omni-formal-core-v1',metadata['core_selection_sha256'])
 core={**metadata,'core_prerequisite_asset':{k:asset[k] for k in ('id','name','bytes','sha256')},'complete_core_revalidated_without_predictions':True}
 expected=runner.job_config('kev',core)
 workers=runner.worker_files();assert len(workers)==62
 output=HERE/'actual-reused-input';output.mkdir()
 job,payload=recovery.reuse_input(expected,output,workers,proof)
 assert len(payload)==434670 and hashlib.sha256(payload).hexdigest()==recovery.IDENTITY['asset_sha256']
 assert job['config_sha256']==recovery.IDENTITY['job_config_sha256']
 assert (output/'input.zip').read_bytes()==Path('/workspace/work/omni-kev-first-preparation-failure-41004743884/input.zip').read_bytes()
 original_job=read(output/'job.json');assert len(original_job['input_files'])==70
 download_calls=[]
 def copy_original(args):download_calls.append(True);(Path(args[-1])/'input.zip').write_bytes(payload)
 for mutation in ('science','worker_set','worker_bytes','frozen_bytes','transport'):
  folder=HERE/('reuse-negative-'+mutation);folder.mkdir();candidate=copy.deepcopy(expected);names=workers[:]
  if mutation=='science':candidate['gpu']='L4'
  if mutation=='worker_set':names=names[:-1]
  real_read=Path.read_bytes;changed=ROOT/workers[0] if mutation=='worker_bytes' else core_inputs.FROZEN/'data-definition.json'
  def mutated_read(self):return b'private modified bytes' if self==changed and mutation in ('worker_bytes','frozen_bytes') else real_read(self)
  def corrupted(args):(Path(args[-1])/'input.zip').write_bytes(payload+b'corrupt')
  try:
   with patch.object(recovery,'command',side_effect=corrupted if mutation=='transport' else copy_original),patch.object(Path,'read_bytes',new=mutated_read):
    recovery.reuse_input(candidate,folder,names,proof)
  except ValueError:pass
  else:raise AssertionError(mutation+' accepted')
  assert not (folder/'input.zip').exists()
  events.append({'case':'reuse_'+mutation,'accepted':False,'private_only':True})
 try:
  with patch.object(recovery,'command',side_effect=copy_original):recovery.reuse_input(expected,output,workers,proof)
 except FileExistsError:pass
 else:raise AssertionError('overwrite allowed')
 assert hashlib.sha256((output/'input.zip').read_bytes()).hexdigest()==recovery.IDENTITY['asset_sha256']
finally:guard.remove()
assert not nn
result={'schema':'omni-mature-runtime-current-recovery-independent-v1','created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'actual_operative_permit_absent_and_eligibility_rejected':missing,'private_eligibility_cases':events,'private_permit_not_operative':True,'actual_predecessor':{'row_sha256':digest(row),'receipt_sha256':digest(row['receipt']),'status':row['status'],'retained_bound_usd':row['reserved_usd'],'retained_round_total_usd':state['reserved_total_usd'],'live_ledger_blob':response['sha']},'actual_readonly_integration':{'committed_metadata6_and_audit57':True,'live_completed_owned_core_asset':{k:asset[k] for k in ('id','name','bytes','sha256')},'real_checked_selection10_models':True,'same_existing_core_evidence_not_fresh_actions_wrapper':True,'reused_original_input_zip_from_live_owned_download':True,'payload_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'job_config_sha256':job['config_sha256'],'manifest_entries':len(original_job['input_files']),'worker_files':len(workers),'job_config_exact_original':True,'nn_forward_attempts':nn},'real_modal_rpc_calls':0,'real_ledger_writes':0,'real_release_writes':0,'operative_permit_created_by_probe':False}
(HERE/'current-recovery-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'all_assertions_passed':True,'eligibility_and_reuse_cases':len(events),'actual_zip_bytes':len(payload),'actual_worker_files':len(workers),'actual_nn_forward_calls':len(nn),'operative_permit_still_absent':not (ROOT/recovery.PERMIT_NAME).exists()}))
