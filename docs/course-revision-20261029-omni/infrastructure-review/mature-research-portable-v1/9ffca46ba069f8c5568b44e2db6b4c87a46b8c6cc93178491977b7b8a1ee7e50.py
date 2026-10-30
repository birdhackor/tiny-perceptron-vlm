"""Offline orchestration failure-order controls; every remote interface is a labeled fixture."""
import copy,hashlib,json,os,sys,types
from pathlib import Path
from unittest.mock import patch
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'original-source';sys.path.insert(0,str(SOURCE))
from scripts.omni_revision import research_execution as rx
from scripts.omni_revision import execution as ex
root=BASE/'probes/original-controller-order';root.mkdir(parents=True,exist_ok=False)
rates={'gpu_hour_cost_l4':'.8','gpu_hour_cost_l40s':'1.95','cpu_hour_cost_sandbox':'.1419','mem_gib_hour_cost_sandbox':'.024','egress_gib_cost':'.04'}
core={'core_selection_sha256':'fixture-selection','core_final_opening_sha256':'fixture-opening','files':{}}
modal=types.ModuleType('modal');modal.__path__=[];modutils=types.ModuleType('modal._utils');modutils.__path__=[];asyncutils=types.ModuleType('modal._utils.async_utils')
results=[]
for mode in ('duplicate_failed_attempt','missing_completed_core','rate_increase','CAS_failure','input_roundtrip_failure','executor_entry_lost','known_stopped_worker_failed'):
 events=[];output=root/mode;ghoutput=root/(mode+'-github-output.txt');ghoutput.write_text('');attempt='mature-kev-999-1'
 prior={'status':'completed','purpose':json.dumps({'job':'cpu-core-formal'}),'receipt':{'success':True,'all_training_processes_exited':True}}
 state={'origin':{'baseline':{'summaries':[{'cycle':'2026-10'}]}},'attempts':[prior],'reserved_total_usd':'2.20'}
 if mode=='duplicate_failed_attempt':state['attempts'].append({'status':'failed-stopped','purpose':json.dumps({'job':'mature-kev'})})
 if mode=='missing_completed_core':prior['status']='failed-stopped'
 used_rates=copy.deepcopy(rates)
 if mode=='rate_increase':used_rates['gpu_hour_cost_l40s']='1.9501'
 modal.Workspace=types.SimpleNamespace(from_context=lambda:types.SimpleNamespace(billing=types.SimpleNamespace(rates=lambda:used_rates)))
 class Store:
  def read(self):events.append('read_ledger');return state,'fixture-observed-blob'
  def write(self,following,blob):
   events.append('CAS')
   if mode=='CAS_failure':raise RuntimeError('fixture CAS outcome unknown; do not create')
   return {'commit':'fixture-commit','blob':'fixture-new-blob'}
 def prepare(job,destination):
  events.append('prepare_non_neural_input');(destination/'input.zip').write_bytes(b'private-fixed-input');job={**job,'config_sha256':'fixture-job'};return job,b'private-fixed-input'
 def reserve(state,snapshot,inventory,attempt,components,revision,purpose):
  events.append('pure_reserve');following=copy.deepcopy(state);following['attempts'].append({'attempt_id':attempt,'status':'reserved','revision':revision,'purpose':purpose,'components_usd':components,'guard':{'retained_official_increment_usd':'0','exact_legacy_carry_usd':'4.73'}});following['reserved_total_usd']='6.50';return following
 def api(*args):events.append('create_owned_draft');return {'id':12}
 def upload(*args):
  events.append('input_roundtrip')
  if mode=='input_roundtrip_failure':raise RuntimeError('fixture remote readback mismatch')
  return {'id':13,'name':'input.zip','bytes':19,'sha256':'fixture-sha','roundtrip_verified':True}
 def execute(*args):
  events.append('enter_modal_executor_fixture')
  if mode=='executor_entry_lost':raise RuntimeError('fixture response lost at executor entry')
  return {'success':False,'paid_call_started':True,'app_id':'ap-fixture-owned','cleanup_checks':{'owned_sandboxes_stopped':True,'client_closed':True},'cleanup_deadline_at':ex.now()}
 asyncutils.synchronizer=types.SimpleNamespace(create_blocking=lambda ignored:execute)
 env={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':rx.REPOSITORY,'GITHUB_REF':'refs/heads/main','GITHUB_EVENT_NAME':'workflow_dispatch','GITHUB_RUN_ID':'999','GITHUB_RUN_ATTEMPT':'1','MODAL_ENVIRONMENT':'main','MODAL_SANDBOX_V2':'0','GITHUB_OUTPUT':str(ghoutput)}
 with patch.dict(os.environ,env),patch.dict(sys.modules,{'modal':modal,'modal._utils':modutils,'modal._utils.async_utils':asyncutils}),patch.object(rx.importlib.metadata,'version',return_value='1.6.0'),patch.object(rx,'checked_sources',return_value={'revision':'1'*40,'files':{'review_fixture':True}}),patch.object(rx,'checked_core',return_value=core),patch.object(rx,'GitHubLedger',Store),patch.object(rx,'prepare',side_effect=prepare),patch.object(rx.preflight,'inventory',return_value={'observed_at':ex.now()}),patch.object(rx.preflight,'load_reconciliation',return_value={}),patch.object(rx.preflight,'official_snapshot',return_value={'observed_at':ex.now()}),patch.object(rx,'official_storage_quote',return_value={'observed_at':ex.now(),'conservative_daily_usd_per_gib':'.02'}),patch.object(rx,'reserve',side_effect=reserve),patch.object(rx,'find_draft',return_value=None),patch.object(rx,'api',side_effect=api),patch.object(rx,'upload_owned',side_effect=upload),patch.object(rx.execution,'read_stopped_apps',return_value=[{'app_id':'ap-fixture-owned','description':'omni-'+attempt,'state':'stopped','tasks':'0'}]):
  try:rx.run('kev',attempt,output)
  except BaseException as error:error_type=type(error).__name__
  else:error_type=None
 receipt=json.loads((output/'research-receipt.json').read_bytes()) if (output/'research-receipt.json').exists() else None
 results.append({'case':mode,'events':events,'error_type':error_type,'receipt_flags':None if receipt is None else {k:receipt.get(k) for k in ('paid_call_started','compute_stopped','resources_verified','success')},'real_remote_calls':0})
 if mode in ('duplicate_failed_attempt','missing_completed_core','rate_increase','CAS_failure','input_roundtrip_failure'):assert 'enter_modal_executor_fixture' not in events
 if 'enter_modal_executor_fixture' in events:assert events.index('CAS')<events.index('input_roundtrip')<events.index('enter_modal_executor_fixture')
 if mode=='executor_entry_lost':assert receipt['paid_call_started'] is True and receipt['compute_stopped'] is False and receipt['resources_verified'] is False
# Actual identity gate separately receives deliberate invalid immutable invocation metadata.
identity=[]
good={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':rx.REPOSITORY,'GITHUB_REF':'refs/heads/main','GITHUB_EVENT_NAME':'workflow_dispatch','GITHUB_RUN_ID':'12','GITHUB_RUN_ATTEMPT':'1','MODAL_ENVIRONMENT':'main','MODAL_SANDBOX_V2':'0'}
for field,value in (('GITHUB_REF','refs/heads/topic'),('GITHUB_EVENT_NAME','push'),('GITHUB_REPOSITORY','other/repo'),('MODAL_ENVIRONMENT','other'),('MODAL_SANDBOX_V2','1')):
 changed={**good,field:value}
 with patch.dict(os.environ,changed):
  try:rx.identity('kev','mature-kev-12-1');accepted=True
  except ValueError:accepted=False
 identity.append({'field':field,'accepted':accepted});assert not accepted
costs={n:rx.cost_proof(rates,rx.job_config(n,core),{'conservative_daily_usd_per_gib':'.02'}) for n in rx.JOBS}
record={'scope':'Actual run/identity/cost/protection functions with explicitly fake sources, core/ledger/reserve/snapshot/Modal/GitHub transports. No remote or neural calls.','orchestration_cases':results,'identity_negative_controls':identity,'costs_at_reviewed_rate_ceilings':costs,'paid_or_model_or_final_calls':0}
p=BASE/'original-controller-order-result.json';p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False,indent=2))
