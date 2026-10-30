"""真 prepare/build 的缺ffmpeg OS負例與本機原生成器正例；所有paid/writer邊界禁止。"""

import copy,hashlib,json,os,shutil,socket,subprocess,sys,wave
from datetime import UTC,datetime
from pathlib import Path
from unittest.mock import patch

ROOT=Path('/workspace/work/omni-models-v1');sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
import modal
from scripts.omni_revision import research_execution as runner,research_cases as cases,research_recovery as recovery
from scripts.omni_revision.finance import digest
def read(p):return json.loads(Path(p).read_bytes())
lr=read(HERE/'provider-ledger.json')['structuredContent'];state=json.loads(lr['content']);assert state['reserved_total_usd']=='10.90'
assert not any(recovery.purpose(row).get('job')=='mature-qwen25' for row in state['attempts'])
old_job=read('/workspace/work/omni-kev-first-preparation-failure-41004743884/job.json')
core={key:old_job[key] for key in ('core_selection_sha256','core_final_opening_sha256','files','core_prerequisite_asset','complete_core_revalidated_without_predictions')}
events=[];ffmpeg_calls=[];real_run=subprocess.run
def forbidden(*args,**kwargs):events.append('forbidden_paid_or_write');raise AssertionError('paid/write boundary reached')
def no_binary(command,*args,**kwargs):
 if isinstance(command,(list,tuple)) and command and command[0]=='ffmpeg':
  ffmpeg_calls.append({'executable':command[0],'filter':command[command.index('-i')+1].split(':voice=')[-1],'timeout':kwargs.get('timeout')})
  return real_run(command,*args,env={'PATH':''},**kwargs)
 return real_run(command,*args,**kwargs)
class Store:
 def read(self):events.append('ledger_read_fixture');return copy.deepcopy(state),lr['sha']
 def write(self,*args,**kwargs):return forbidden(*args,**kwargs)
environment={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':'birdhackor/tiny-perceptron-vlm','GITHUB_REF':'refs/heads/main','GITHUB_EVENT_NAME':'workflow_dispatch','GITHUB_RUN_ID':'41012209461','GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':'010a639f6db4889ecbac5801dcab7a7a4409e8f0','MODAL_ENVIRONMENT':'main','MODAL_SANDBOX_V2':'0'}
output=HERE/'missing-ffmpeg-controller-private';error=None
with patch.dict(os.environ,environment),patch.object(runner,'GitHubLedger',Store),patch.object(runner,'checked_core',return_value=core),patch.object(subprocess,'run',side_effect=no_binary),patch.object(modal.Workspace,'from_context',side_effect=forbidden),patch.object(runner.preflight,'inventory',side_effect=forbidden),patch.object(runner,'official_storage_quote',side_effect=forbidden),patch.object(runner,'reserve',side_effect=forbidden),patch.object(runner,'api',side_effect=forbidden),patch.object(socket.socket,'connect',side_effect=forbidden):
 try:runner.run('qwen25','mature-qwen25-41012209461-1',output)
 except FileNotFoundError as exc:error={'type':type(exc).__name__,'filename':exc.filename,'message':str(exc)}
 else:raise AssertionError('missing ffmpeg accepted')
assert error['filename']=='ffmpeg' and events==['ledger_read_fixture'] and len(ffmpeg_calls)==1
assert (output/'sources.json').exists() and not any((output/n).exists() for n in ('reservation.json','input.zip','job.json','research-receipt.json'))
assert recovery.eligibility(state,'qwen25') is None
try:recovery.eligibility(state,'kev')
except ValueError:kev_closed=True
else:raise AssertionError('used Kev recovery allowed')
binary=shutil.which('ffmpeg');assert binary
positive=HERE/'actual-local-generator-cases';built=cases.build(positive)
assert len(built['audio_text_pairs'])==24 and len(built['av_shift_pairs'])==24 and len(built['speech_output_prompts'])==8
assert built['model_outputs_used'] is False and built['human_listening_test'] is False
audio_summary=[]
for row in built['audio_text_pairs']:
 path=positive/row['audio']
 with wave.open(str(path),'rb') as source:
  rate=source.getframerate();channels=source.getnchannels();width=source.getsampwidth();duration=source.getnframes()/rate
 assert rate==16000 and channels==1 and width==2 and 0<duration<=8
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['audio_sha256']
 audio_summary.append({'id':row['id'],'bytes':path.stat().st_size,'sha256':row['audio_sha256'],'sample_rate':rate,'channels':channels,'sample_width':width,'duration':duration})
for row in built['av_shift_pairs']:
 assert len(row['frames'])==8 and len(row['pair'])==2 and row['fps']==2.0 and row['duration_seconds']==4
result={'schema':'omni-mature-controller-media-original-independent-probe-v1','created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'missing_binary_case':{'real_original_run_prepare_build':True,'OS_binary_lookup_control':'only ffmpeg subprocess inherits PATH empty; no handwritten FileNotFoundError','actual_error':error,'ffmpeg_calls':ffmpeg_calls,'paid_or_write_boundary_events':events,'reservation_or_input_archive_created':False,'mocked_readonly_boundaries':['ledger read returns fresh genuine recorded ledger','strict core returns genuine original frozen core metadata to focus on prepare failure']},'local_actual_generator_positive':{'ffmpeg_path':binary,'ffmpeg_version':built['ffmpeg_version'],'voice':'flite/slt','audio_pairs':24,'av_pairs':24,'speech_output_prompts':8,'audio':audio_summary,'frames_per_av_pair':8,'fps':2.0,'model_outputs_used':False,'case_manifest_bytes':(positive/'cases.json').stat().st_size,'case_manifest_sha256':hashlib.sha256((positive/'cases.json').read_bytes()).hexdigest()},'finance_once':{'ledger_blob':lr['sha'],'retained_total_usd':'10.90','qwen_has_no_prior_reserved_attempt':True,'used_sole_kev_permit_remains_closed':kev_closed},'real_modal_rpc_calls':0,'real_ledger_writes':0,'real_release_writes':0,'nn_predictions':0,'limitations':['The complete generator positive uses this existing local ffmpeg; it does not establish that a future clean Actions runner installs ffmpeg or contains flite/slt.','No actual Kev result/capability is reviewed. Ledger completion is cited only for permit consumption and retained bounds.']}
(HERE/'media-prerequisite-probe-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'all_assertions_passed':True,'missing_binary_before_paid':error['filename'],'actual_local_generated_audio_pairs':24,'av_pairs':24,'speech_prompts':8,'qwen_paid_rows':0,'real_modal_rpc_calls':0}))
