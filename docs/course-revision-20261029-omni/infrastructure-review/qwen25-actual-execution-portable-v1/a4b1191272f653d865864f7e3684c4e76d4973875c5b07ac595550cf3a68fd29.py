"""真Qwen付費失敗：原生GET、來源/input/預約/成本/停止唯讀核對；不呼叫NN或遠端writer。"""

import ast,copy,hashlib,io,json,math,os,struct,subprocess,sys,wave,zipfile
from datetime import UTC,datetime,timedelta
from decimal import Decimal,ROUND_CEILING
from pathlib import Path
from unittest.mock import patch

ROOT=Path('/workspace/work/omni-book-integrated-v1');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
ACTUAL=Path('/workspace/work/omni-qwen25-original-attempt-41015365689');A=ACTUAL/'artifact'
import torch
from PIL import Image
from scripts.omni_revision import research_execution as runner,research_recovery as recovery,core_final
from scripts.omni_revision.finance import digest,reservation_identity,validate_state,finish,resources_verified
from tiny_perceptron.omni_products.training_runtime import source_binding

def read(p):return json.loads(Path(p).read_bytes())
def sha(raw):return hashlib.sha256(raw).hexdigest()
def binding(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':sha(raw)}
def utc(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
def retained_fields(expected,actual):
 if isinstance(expected,dict):return isinstance(actual,dict) and all(k in actual and retained_fields(v,actual[k]) for k,v in expected.items())
 return expected==actual
def ceilcent(v):return v.quantize(Decimal('.01'),rounding=ROUND_CEILING)

started=datetime.now(UTC).isoformat();REV='99baece1f20e5758dfdae9cdccdadb2d600f6029';ATTEMPT='mature-qwen25-41015365689-1'
run=json.loads(read(HERE/'provider-run.json')['structuredContent']['content']);jobs=read(HERE/'provider-jobs.json')['structuredContent']['jobs'];artifacts=read(HERE/'provider-artifacts.json')['structuredContent']['artifacts']
assert run['id']==41015365689 and run['run_attempt']==1 and run['head_sha']==REV and run['head_branch']=='main' and run['event']=='workflow_dispatch' and run['status']=='completed' and run['conclusion']=='failure'
assert len(jobs)==1 and jobs[0]['id']==123930385331 and jobs[0]['conclusion']=='failure'
steps={x['name']:x['conclusion'] for x in jobs[0]['steps']}
assert steps['Install and verify Qwen input speech preparation before reservation']=='success'
assert steps['Verify the real Modal filesystem contract offline before reservation']=='success'
assert steps['Prepare and reserve once, infer, export and stop the exact owned resources']=='failure'
for name in ('Check all evidence size before artifact upload','Retain complete or partial original evidence for seven days','Verify provider artifact limits and finish the exact reservation without refund'):assert steps[name]=='success'
assert len(artifacts)==1;artifact=artifacts[0];assert artifact['id']==12959736567 and artifact['size_in_bytes']==2582922 and artifact['digest']=='sha256:d646a8743e84c0434a32aac80ddbd59876b71d5bc5152e4b1adcf07cdee02380' and not artifact['expired']
assert artifact['workflow_run']['head_sha']==REV
retention=utc(artifact['expires_at'])-utc(artifact['created_at']);assert timedelta(days=6,hours=23)<retention<=timedelta(days=7)
original_zip=ACTUAL/'original-provider-artifact.zip';assert binding(original_zip)['bytes']==artifact['size_in_bytes'] and 'sha256:'+binding(original_zip)['sha256']==artifact['digest']
entries=[]
with zipfile.ZipFile(original_zip) as archive:
 assert len(archive.infolist())==len(set(archive.namelist()))==18
 for info in archive.infolist():
  assert not info.filename.startswith('/') and '..' not in Path(info.filename).parts
  raw=archive.read(info);assert (A/info.filename).read_bytes()==raw
  entries.append({'path':info.filename,'bytes':len(raw),'sha256':sha(raw)})
assert set(archive.namelist())=={p.name for p in A.iterdir() if p.is_file()}
assert not any((A/n).exists() for n in ('result.zip','result-summary.json','stdout.txt','worker-result.json'))

lr=read(HERE/'provider-ledger.json')['structuredContent'];ledger=json.loads(lr['content']);blobbytes=lr['content'].encode();assert hashlib.sha1(b'blob '+str(len(blobbytes)).encode()+b'\0'+blobbytes).hexdigest()==lr['sha']
qrows=[x for x in ledger['attempts'] if recovery.purpose(x).get('job')=='mature-qwen25'];assert len(qrows)==1
row=qrows[0];assert row['attempt_id']==ATTEMPT and row['status']=='failed-stopped' and row['reserved_usd']=='3.10' and row['revision']==REV
assert ledger['reserved_total_usd']=='14.00' and sum(Decimal(x['reserved_usd']) for x in ledger['attempts'])==Decimal('14.00')
receipt=row['receipt'];assert receipt['success'] is False and receipt['compute_stopped'] is True and receipt['resources_verified'] is True and resources_verified(receipt)
assert receipt['attempt']==ATTEMPT and receipt['job']=='qwen25' and receipt['schema']=='omni-mature-controller-receipt-v2' and receipt['source_revision']==REV
assert receipt['phase']=='waiting_for_task_id' and receipt['error_type']=='TimeoutError' and receipt['paid_call_started'] is True
assert receipt['preparation_recovery'] is None
assert all(receipt['cleanup_checks'].values()) and receipt['stop_errors']=={} and receipt['remaining_owned_sandboxes']==[]
assert receipt['observed_owned_sandbox_ids']==['sb-JIbsKnP9QzUS4ddAUbYrdc'] and receipt['stopped_sandboxes']=={'sb-JIbsKnP9QzUS4ddAUbYrdc':3}
assert receipt['app_id']=='ap-LwpbDVjG2JbIqQzGBWuLAw' and receipt['sandbox_id']=='sb-JIbsKnP9QzUS4ddAUbYrdc'
runtime=read(A/'runtime-receipt.json');progress=read(A/'runtime-progress.json');research=read(A/'research-receipt.json');reserved_file=read(A/'reservation.json');reserved=reserved_file['attempt']
assert retained_fields(runtime,receipt)
assert retained_fields(research,receipt)
assert progress['phase']=='waiting_for_task_id' and progress['sandbox_id']==receipt['sandbox_id']
assert all(k not in runtime for k in ('input_payload_uploaded','result_exported_before_stop','research_final_opened'))
assert reservation_identity(row)==reservation_identity(reserved) and reserved['status']=='reserved' and reserved['reservation_sha256']==digest(reservation_identity(reserved))
early_raw=read(HERE/'provider-reservation-blob.json')['structuredContent']['content'].encode();early_blob=hashlib.sha1(b'blob '+str(len(early_raw)).encode()+b'\0'+early_raw).hexdigest();assert early_blob==reserved_file['transaction']['blob']=='60cfe7a8dce887fa19f2c315f882fa74a0d17f1c'
early=json.loads(early_raw);assert next(x for x in early['attempts'] if x['attempt_id']==ATTEMPT)==reserved
commit=json.loads(read(HERE/'provider-reservation-commit.json')['structuredContent']['content']);assert commit['sha']==reserved_file['transaction']['commit']=='ec1043cc524fd73f3c36615ded4e7449c8e6f8fb'
assert utc(commit['committer']['date'])<=utc(receipt['paid_call_at']) and utc(reserved['reserved_at'])<=utc(receipt['paid_call_at'])
validate_state(ledger,datetime.now(UTC));validate_state(early,datetime.now(UTC))
replayed=finish(early,ATTEMPT,copy.deepcopy(receipt));assert next(x for x in replayed['attempts'] if x['attempt_id']==ATTEMPT)['status']=='failed-stopped' and replayed['reserved_total_usd']==early['reserved_total_usd']=='14.00'
unknown=copy.deepcopy(receipt);unknown['cleanup_checks']['owned_sandboxes_stopped']=False;unresolved=finish(early,ATTEMPT,unknown);assert next(x for x in unresolved['attempts'] if x['attempt_id']==ATTEMPT)['status']=='unresolved'

release=read(HERE/'provider-release-gh.json');connector=json.loads(read(HERE/'provider-release-connector.json')['structuredContent']['content'])
for k in ('id','tag_name','target_commitish','draft','body'):assert release[k]==connector[k]
assert release['id']==receipt['release_id']==428722920 and release['draft'] is True and release['published_at'] is None
tag,body=runner.release_identity(ATTEMPT,REV);assert release['tag_name']==receipt['release_tag']==tag and release['body']==receipt['release_body']==body and release['target_commitish']==REV
assert len(release['assets'])==len(receipt['assets'])==1;asset=release['assets'][0];expected_asset=receipt['assets'][0]
assert asset['id']==expected_asset['id']==691208249 and asset['name']=='input.zip' and asset['size']==expected_asset['bytes']==2610453 and asset['digest']=='sha256:'+expected_asset['sha256']
assert expected_asset['roundtrip_verified'] is True
payload=(A/'input.zip').read_bytes();assert len(payload)==asset['size'] and sha(payload)==expected_asset['sha256']==receipt['input_payload_sha256']=='06e03d9e4bfe5ce8afaf9b7c99f3e6396c20c77c1bea6c014c1ff9b4d1ddc65f'
sources=read(A/'sources.json');audit_path=ROOT/'docs/course-revision-20261029-omni/infrastructure/mature-research-audit-v4.json';audit=read(audit_path)
assert sha(audit_path.read_bytes())==sources['audit_sha256']==receipt['controller_audit_sha256']=='a9dff64302eb183cce247765f88b98d16e6ed6b8a8a7ee8257401a770bab37f8'
assert len(sources['files'])==96 and sources['files']==audit['files'] and sources['revision']==REV
nn_attempts=[]
def no_forward(module,inputs):nn_attempts.append(type(module).__name__);raise AssertionError('no NN in actual failure review')
hook=torch.nn.modules.module.register_module_forward_pre_hook(no_forward)
try:
 with patch.dict(os.environ,{'GITHUB_SHA':REV}):assert runner.checked_sources()==sources
 source_rows=[]
 for name,h in sources['files'].items():
  actual=(ROOT/name).read_bytes();committed=subprocess.check_output(['git','show',REV+':'+name],cwd=ROOT);assert actual==committed and sha(actual)==h
  source_rows.append({'path':name,'sha256':h,'bytes':len(actual)})
 runner.check_receipt_identity('qwen25',ATTEMPT,sources,reserved,receipt)
 actual_bounds=runner.bound(A);assert actual_bounds==receipt['artifact']
 eval_sources=core_final.evaluation_sources();assert len(eval_sources)==57
 for name,h in eval_sources.items():assert sha(subprocess.check_output(['git','show',REV+':'+name],cwd=ROOT))==h
 recipe=read(ROOT/'docs/omni-products/v1/formal-recipes.json');assert len(recipe['execution_sources_sha256'])==25
 for name,h in recipe['execution_sources_sha256'].items():assert sha((ROOT/name).read_bytes())==h
 assert source_binding()==recipe['implementation_sha256'] and len(source_binding())==8
 assert len(runner.worker_files())==62
 once=[]
 for job in ('qwen25','kev'):
  try:recovery.eligibility(ledger,job)
  except ValueError:once.append({'job':job,'future_attempt_rejected':True})
  else:raise AssertionError(job+' retry became eligible')
finally:hook.remove()
assert not nn_attempts

jobraw=(A/'job.json').read_bytes();job=json.loads(jobraw);purpose=json.loads(row['purpose']);assert sha(jobraw)==purpose['job_config_sha256']==receipt['config_sha256']=='b38df87cdece704549804679462c493aead4c7bd61936d41acf24387250fa5bc'
assert purpose['job']=='mature-qwen25' and purpose['sources_sha256']==digest(sources) and purpose['controller_audit_sha256']==sources['audit_sha256']
assert purpose['input_sha256']==sha(payload) and not any(k in purpose for k in recovery.LINEAGE_KEYS)
assert job['job']=='qwen25' and job['gpu']=='L4' and job['cpu']==4 and job['memory_mib']==49152 and job['sandbox_seconds']==2100 and job['worker_seconds']==1800 and job['own_training_permitted'] is False
assert job['image_id']=='im-1b8GW96gjEVRiK1vg5Jqa0'
metadata=[];assert len(job['files'])==6
for name,h in job['files'].items():
 path=ROOT/'docs/omni-products/v1'/name;raw=path.read_bytes();assert sha(raw)==h and subprocess.check_output(['git','show',REV+':'+str(path.relative_to(ROOT))],cwd=ROOT)==raw;metadata.append({'name':name,'sha256':h,'bytes':len(raw)})
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
 infos=bundle.infolist();assert len(infos)==len(set(bundle.namelist()))==355 and sum(x.file_size for x in infos)<=64*1024**2
 assert bundle.read('job.json')==jobraw and set(bundle.namelist())==set(job['input_files'])|{'job.json'} and len(job['input_files'])==354
 input_entries=[]
 for name,bound in job['input_files'].items():
  assert not name.startswith('/') and '..' not in Path(name).parts and '\\' not in name
  raw=bundle.read(name);assert len(raw)==bound['bytes'] and sha(raw)==bound['sha256'];input_entries.append({'path':name,**bound})
 workers=runner.worker_files();assert {n for n in job['input_files'] if n.startswith(('scripts/','tiny_perceptron/'))}==set(workers)
 for name in workers:assert bundle.read(name)==(ROOT/name).read_bytes()
 for name in ('data-definition.json','model-selection.json','final-opening.json'):assert bundle.read(name)==(ROOT/'docs/omni-products/v1'/name).read_bytes()
 cases=json.loads(bundle.read('cases/cases.json'));assert cases['schema']=='omni-mature-research-cases-v1' and cases['model_outputs_used'] is False and cases['human_listening_test'] is False
 assert cases['source_sha256']==sources['files']['scripts/omni_revision/research_cases.py']
 assert (len(cases['audio_text_pairs']),len(cases['av_shift_pairs']),len(cases['speech_output_prompts']))==(24,24,8)
 tree=ast.parse((ROOT/'scripts/omni_revision/research_cases.py').read_bytes());const={x.targets[0].id:ast.literal_eval(x.value) for x in tree.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id in ('PROMPTS','SPEECH_WORDS')}
 audio=[]
 for record,(text,gold) in zip(cases['audio_text_pairs'],const['PROMPTS'],strict=True):
  assert record['transcript']==text and record['gold']==gold;raw=bundle.read('cases/'+record['audio']);assert sha(raw)==record['audio_sha256']
  with wave.open(io.BytesIO(raw),'rb') as w:
   rate=w.getframerate();channels=w.getnchannels();width=w.getsampwidth();frames=w.getnframes();samples=w.readframes(frames)
  assert rate==16000 and channels==1 and width==2 and 0<frames/rate<=8 and any(samples)
  assert math.isclose(frames/rate,record['duration_seconds'],rel_tol=1e-12)
  audio.append({'id':record['id'],'bytes':len(raw),'sha256':sha(raw),'sample_rate':rate,'channels':channels,'sample_width':width,'duration_seconds':frames/rate})
 av=[]
 for record in cases['av_shift_pairs']:
  assert len(record['frames'])==8 and len(record['pair'])==2 and record['fps']==2.0 and record['duration_seconds']==4
  bits=[(record['abstract_pattern']>>(7-i))&1 for i in range(8)]
  for i,frame in enumerate(record['frames']):
   raw=bundle.read('cases/'+frame['path']);assert sha(raw)==frame['sha256'] and frame['timestamp_ms']==i*500
   with Image.open(io.BytesIO(raw)) as im:assert im.mode=='RGB' and im.size==(112,112) and im.getpixel((56,56))==((220,20,20) if bits[i]==0 else (20,20,220))
  for color,pair in enumerate(record['pair']):
   assert pair['gold']==('red' if color==0 else 'blue') and pair['beep_center_seconds']==next(i for i,v in enumerate(bits) if v==color)*.5+.25
   raw=bundle.read('cases/'+pair['audio']);assert sha(raw)==pair['sha256']
   with wave.open(io.BytesIO(raw),'rb') as w:assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,64000)
  av.append({'id':record['id'],'frames':8,'size':[112,112],'fps':2.0,'duration_seconds':4,'pair_count':2})
 for record,word in zip(cases['speech_output_prompts'],const['SPEECH_WORDS'],strict=True):assert record['text']=='Say only the word '+word+'.' and record['gold_word']==word

proof=read(A/'cost-proof.json');billing=read(A/'billing-before.json');inventory=read(A/'inventory.json');protection=read(A/'remaining-protection.json')
assert purpose['cost_proof_sha256']==digest(proof) and proof['reserved_usd']=='3.10' and proof['components_usd']==row['components_usd']
rates=proof['rates'];hourly=Decimal(rates['gpu_hour_cost_l4'])+4*Decimal(rates['cpu_hour_cost_sandbox'])+48*Decimal(rates['mem_gib_hour_cost_sandbox']);assert hourly==Decimal(proof['hourly_resource_bound_usd'])==Decimal('2.519600')
assert proof['resource_wall_bound_seconds']==2100+480+120==2700
assert ceilcent(hourly*Decimal(2100)/3600)<=Decimal(row['components_usd']['compute']) and ceilcent(hourly*Decimal(600)/3600)<=Decimal(row['components_usd']['startup_idle'])
assert sum(Decimal(x) for x in row['components_usd'].values())==Decimal('3.10')
assert Decimal(row['components_usd']['storage_cleanup'])>=Decimal(proof['storage']['artifact_storage_bound_usd'])+Decimal(proof['storage']['other_cleanup_bound_usd'])
assert ceilcent(Decimal('.25')*Decimal(7)*Decimal('.02'))==Decimal('.04')
assert proof['storage']['artifact_max_bytes']==256*1024**2 and proof['storage']['artifact_retention_days']==7
assert actual_bounds['control_bytes']<=16*1024**2 and actual_bounds['total_bytes']<=256*1024**2 and len(payload)<=64*1024**2 and len(payload)<=128*1024**2
assert not proof['new_persistent_storage'] and not proof['image_build_permitted'] and not proof['automatic_retry_or_gpu_upgrade']
for record in (proof,billing,inventory,proof['storage']['official_quote']):assert 0<=(utc(receipt['paid_call_at'])-utc(record['observed_at'])).total_seconds()<=120
assert protection['future_bounds_usd']=='2.00' and Decimal(protection['guarded_total_usd'])<=Decimal('60')
assert Decimal(protection['guarded_total_usd'])==Decimal(row['guard']['retained_official_increment_usd'])+2*(Decimal(ledger['reserved_total_usd'])+Decimal(protection['future_bounds_usd']))+Decimal(row['guard']['exact_legacy_carry_usd'])
assert runner.protect_remaining(early,'qwen25')==protection
elapsed=(utc(runtime['cleanup_started_at'])-utc(runtime['paid_call_at'])).total_seconds();cleanup=(utc(runtime['finished_at'])-utc(runtime['cleanup_started_at'])).total_seconds()
assert 479.9<=elapsed<=481 and 0<=cleanup<120 and utc(runtime['finished_at'])<=utc(runtime['cleanup_deadline_at'])
apps=read(A/'apps-after.json');owned=[x for x in apps if x['app_id']==receipt['app_id']];assert len(owned)==1 and owned[0]['state']=='stopped' and owned[0]['tasks']=='0' and owned[0]['description']=='omni-'+ATTEMPT
assert utc(owned[0]['stopped_at'])<=utc(runtime['cleanup_deadline_at'])
logcomparison=read(HERE/'native-log-comparison.json');assert logcomparison['provided_has_one_extra_final_LF'] and logcomparison['provided_file_untouched']
cli_stderr=(HERE/'modal-live-apps.stderr').read_bytes();assert b'Token missing' in cli_stderr or b'token missing' in cli_stderr

author_path=ROOT/'docs/omni-products/v1/qwen25-research-summary.json';author=read(author_path)
assert author['run_id']==run['id'] and author['budget']['attempt_reserved_usd']=='3.10' and author['budget']['round_retained_total_usd']=='14.00' and author['actual_execution']['last_phase']==receipt['phase']
assert author['prepared_inputs']['payload']['sha256']==sha(payload) and author['actual_execution']['accuracy'] is None and author['actual_execution']['latency_seconds'] is None and author['actual_execution']['peak_GPU_memory_bytes'] is None

result={'schema':'omni-qwen-actual-bounded-failure-independent-verification-v1','started_at':started,'finished_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'run':{k:run[k] for k in ('id','run_attempt','head_sha','head_branch','status','conclusion','created_at','updated_at','html_url')},'job_id':jobs[0]['id'],'steps':steps,'active_audit':binding(audit_path),'source96':source_rows,'source_scopes':{'execution_sources':96,'worker_sources':62,'evaluation_sources':57,'training_sources':25,'implementation_sources':8,'all_match_paid_commit_and_current':True,'real_checked_sources_and_receipt_identity_positive':True},'artifact':{k:artifact[k] for k in ('id','name','size_in_bytes','digest','created_at','expires_at','expired')},'artifact18_entries':entries,'payload':{'bytes':len(payload),'sha256':sha(payload),'zip_entries':355,'expanded_bytes':sum(x.file_size for x in infos),'manifest_files':354,'worker_files':62,'job_config_sha256':sha(jobraw),'all_manifest_sha_size_and_workers_verified':True,'six_metadata':metadata},'prepared_cases':{'audio_text_pairs':24,'av_shift_pairs':24,'speech_output_prompts':8,'planned_qwen_generate_calls':104,'planned_asr_proxy_calls':8,'actual_ffmpeg_input_preparation_version':cases['ffmpeg_version'],'audio_pcm_checks':audio,'av_image_wave_contracts':av,'these_are_inputs_not_model_outputs':True},'finance':{'ledger_blob':lr['sha'],'terminal_status':row['status'],'reservation_components':row['components_usd'],'attempt_bound_retained_usd':'3.10','all_round_bounds_retained_usd':'14.00','reserved_before_paid_commit':commit['sha'],'reserved_blob':early_blob,'reservation_identity_exact':True,'reserved_before_paid_call':True,'historical_rate_hourly_resource_bound_usd':str(hourly),'full_resource_wall_bound_seconds':2700,'compute_lower_required_bound_usd':str(ceilcent(hourly*Decimal(2100)/3600)),'startup_cleanup_lower_required_bound_usd':str(ceilcent(hourly*Decimal(600)/3600)),'recorded_storage_ceiling_bound_usd':'.04','future_bounds_usd':protection['future_bounds_usd'],'guarded_total_usd':protection['guarded_total_usd'],'future_guard_formula':'retained_official_increment + 2*(all_new_bounds+future_bounds) + exact_legacy_carry','base_gross_guarded_increment_usd':row['guard']['guarded_increment_usd'],'cap_usd':'60','historical_snapshots_fresh_at_paid_call':True,'recorded_artifact_bounds':actual_bounds,'pure_finish_replay':'failed-stopped; all bounds unchanged','incomplete_cleanup_control':'unresolved','bound_is_actual_invoice':False},'draft_release':{k:release[k] for k in ('id','tag_name','target_commitish','draft','body','published_at')},'only_release_asset':{k:asset[k] for k in ('id','name','size','digest','state')},'failure':{'paid_call_at':runtime['paid_call_at'],'last_phase':runtime['phase'],'error_type':runtime['error_type'],'cleanup_started_at':runtime['cleanup_started_at'],'runtime_finished_at':runtime['finished_at'],'shared_create_and_task_id_elapsed_seconds':elapsed,'cleanup_elapsed_seconds':cleanup,'deadline_seconds':480,'SandboxCreate_response_ID_observed':True,'task_ID_observed_in_retained_records':False,'controller_payload_upload_success_recorded':False,'controller_payload_upload_branch_reached_by_fixed_source':False,'controller_sha_marker_send_branch_reached_by_fixed_source':False,'source_basis':'staged source only enters upload after _get_task_id await returns; terminal phase remains waiting_for_task_id. Shared480 deadline covers create RPC plus task-ID wait; no per-subphase duration inferred.','provider_root_cause':'not established','model_inference_or_capability_evaluation_completed':False,'raw_worker_logits_or_generation_results_present':False,'result_zip_present':False,'actual_GPU_type_runtime_observed':None,'GPU_OOM_established':False,'actual_model_dtype':None,'actual_accuracy':None,'actual_model_latency':None},'cleanup':{'app_id':receipt['app_id'],'sandbox_id':receipt['sandbox_id'],'observed_owned_sandbox_ids':receipt['observed_owned_sandbox_ids'],'stopped_sandboxes':receipt['stopped_sandboxes'],'remaining_owned_sandboxes':receipt['remaining_owned_sandboxes'],'stop_errors':receipt['stop_errors'],'checks':receipt['cleanup_checks'],'historical_official_app_projection':owned[0],'finish_before_deadline':True,'fresh_Modal_query':{'exit_code':1,'token_missing':True,'state_refreshed':False,'stderr_fingerprint':binding(HERE/'modal-live-apps.stderr')}},'once_guards':once,'author_summary_consistency':{'reference':binding(author_path),'role':'author summary cross-check only; not authority','checked_observed_fields_consistent':True,'independent_pending_field_not_modified':True},'native_log_comparison':logcomparison,'reviewer_actions':{'NN_forward_or_generation_calls':len(nn_attempts),'paid_compute_calls':0,'dispatches':0,'ledger_writes':0,'release_writes':0,'model_or_recipe_source_edits':0,'commit':False},'limits':['Historical provider stop acknowledgements plus artifact App inventory are verified; no fresh authenticated Modal inventory is available.','The480s interval is shared startup/create/task-ID deadline, not a model inference latency measurement. Provider scheduling/bootstrap/CUDA/OOM cause is not observed.','Source-derived pre-upload path exclusion is separate from direct provider neural-execution telemetry, which is unavailable. No measured model capability exists.','Actual input/case preparation is verified; no processor/model/speech-generation/Whisper outputs are available. Planned104+8 must not be reported as executed per-case failures.','No new Qwen or Kev attempt is authorized; both once guards reject current completed/failed history.']}
(HERE/'actual-failure-verification-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
(HERE/'input-entry-manifest.json').write_text(json.dumps({'payload_sha256':sha(payload),'entries':input_entries},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'all_assertions_passed':True,'sources':96,'workers':62,'input_entries':355,'metadata':6,'execution':'failed-stopped','bound_retained_usd':'3.10','total_retained_usd':'14.00','startup_deadline_elapsed':elapsed,'reviewer_NN':0,'retry_authorized':False}))
