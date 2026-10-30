"""執行精確抽出的workflow Python；真本機FFmpeg正例/OS負例、PCM拒絕及shell安裝失敗阻擋。"""
import contextlib,hashlib,io,json,os,shutil,subprocess,sys,wave
from datetime import UTC,datetime
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;ROOT=Path('/workspace/work/omni-models-v1')
code=(HERE/'workflow-python-extracted.py').read_text();script=(HERE/'workflow-step-extracted.sh').read_text();wf=(ROOT/'.github/workflows/omni-mature-research.yml').read_text()
start=wf.index('      - name: Install and verify Qwen input speech preparation before reservation');end=wf.index('      - name: Verify the real Modal filesystem contract offline before reservation',start);step=wf[start:end]
assert "if: inputs.job == 'qwen25'" in step and 'timeout-minutes: 10' in step and 'env:' not in step and 'secrets.' not in step and 'continue-on-error' not in wf
assert start<wf.index('      - name: Verify the real Modal filesystem contract offline before reservation')<wf.index('      - name: Prepare and reserve once, infer, export and stop the exact owned resources')
assert script.startswith('sudo apt-get update -qq\nsudo apt-get install -y --no-install-recommends ffmpeg\n')
real_run=subprocess.run;cases=[];actual_audio=[]
def execute(name,handler,expect_error):
 stdout=io.StringIO();error=None
 with patch.object(subprocess,'run',side_effect=handler),contextlib.redirect_stdout(stdout):
  try:exec(compile(code,'<exact-extracted-workflow-python>','exec'),{})
  except (OSError,AssertionError,EOFError,wave.Error,subprocess.SubprocessError) as exc:error=type(exc).__name__
 assert (error is not None)==expect_error,(name,error)
 cases.append({'case':name,'rejected':error is not None,'error_type':error,'success_message':stdout.getvalue().strip()})
def actual(command,**kwargs):
 assert command[0]=='ffmpeg' and command[command.index('-i')+1]=='flite=text=ready:voice=slt'
 assert command[command.index('-t')+1]=='0.25' and kwargs=={'check':True,'capture_output':True,'timeout':30}
 result=real_run(command,**kwargs);actual_audio.append(result.stdout);return result
execute('actual_local_ffmpeg_flite_slt',actual,False)
audio=actual_audio[0]
with wave.open(io.BytesIO(audio),'rb') as w:
 sample_data=w.readframes(16000);meta={'channels':w.getnchannels(),'sample_width':w.getsampwidth(),'sample_rate':w.getframerate(),'actual_frames_read':len(sample_data)//(w.getnchannels()*w.getsampwidth()),'nonzero':any(sample_data)}
assert meta=={'channels':1,'sample_width':2,'sample_rate':16000,'actual_frames_read':4000,'nonzero':True}
def missing(command,**kwargs):return real_run(command,env={'PATH':''},**kwargs)
execute('OS_missing_ffmpeg',missing,True)
for name,value in [('real_missing_filter','no_such_filter_for_independent_review=text=ready'),('real_missing_voice','flite=text=ready:voice=no_such_voice_for_independent_review')]:
 def bad_filter(command,**kwargs):
  altered=command[:];altered[altered.index('-i')+1]=value;return real_run(altered,**kwargs)
 execute(name,bad_filter,True)
def wav(channels=1,width=2,rate=16000,frames=1000,silent=False):
 buffer=io.BytesIO()
 with wave.open(buffer,'wb') as w:
  w.setnchannels(channels);w.setsampwidth(width);w.setframerate(rate);w.writeframes((b'\0' if silent else b'\1')*(channels*width*frames))
 return buffer.getvalue()
for name,payload in [('empty_bytes',b''),('stereo',wav(channels=2)),('not_pcm16',wav(width=1)),('wrong_sample_rate',wav(rate=8000)),('silent_audio',wav(silent=True)),('too_short',wav(frames=10))]:
 def fake_output(command,**kwargs):return subprocess.CompletedProcess(command,0,stdout=payload,stderr=b'')
 execute('validation_'+name,fake_output,True)
def timeout(command,**kwargs):raise subprocess.TimeoutExpired(command,30)
execute('subprocess_deadline_propagates',timeout,True)
# 只執行帶fake sudo的shell：首安裝命令exit7即停止，未執行apt或Python。
private=HERE/'install-failure-private-bin';private.mkdir();marker=HERE/'unexpected-python-marker'
(private/'sudo').write_text('#!/bin/sh\nexit 7\n');(private/'sudo').chmod(0o755)
(private/'python').write_text('#!/bin/sh\n/usr/bin/touch '+str(marker)+'\nexit 99\n');(private/'python').chmod(0o755)
bash=shutil.which('bash');failed=real_run([bash,'-e',str(HERE/'workflow-step-extracted.sh')],env={'PATH':str(private)},capture_output=True,timeout=5)
assert failed.returncode==7 and not marker.exists()
version=subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]
result={'schema':'omni-mature-media-workflow-current-independent-probe-v1','created_at':datetime.now(UTC).isoformat(),'reviewer_task':'/root/omni_development_execution_review','all_assertions_passed':True,'exact_workflow_python_sha256':hashlib.sha256(code.encode()).hexdigest(),'workflow_script_syntax_checked':True,'workflow_order_and_scope':{'qwen_only':True,'step_timeout_minutes':10,'no_secret_env_on_setup':True,'before_sdk_and_prepare_reservation':True,'no_continue_on_error':True},'actual_local_ffmpeg':{'path':shutil.which('ffmpeg'),'version':version,'audio_bytes':len(audio),'audio_sha256':hashlib.sha256(audio).hexdigest(),**meta},'cases':cases,'install_failure_shell_control':{'fake_sudo_exit':failed.returncode,'python_not_reached':not marker.exists(),'real_apt_commands_executed':0},'real_actions_installation_verified':False,'actual_nn_predictions':0,'real_modal_rpc_calls':0,'real_ledger_or_release_writes':0,'limitations':['Positive uses existing local FFmpeg7.1.5, not an Actions apt install.','Format/silence/empty/timeout are explicit private subprocess output fixtures; missing executable/filter/voice are real OS/FFmpeg failures.','Frame output is bounded by the exact -t0.25 command and timeout30; wave pipe length metadata is not treated as final duration.']}
(HERE/'workflow-media-gate-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'all_assertions_passed':True,'media_cases':len(cases),'actual_pcm_frames':meta['actual_frames_read'],'install_failure_blocks_python':True,'Actions_apt_verified':False,'paid_calls':0}))
