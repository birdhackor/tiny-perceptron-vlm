"""Cold preparation through two real bounded children, real Hub HTTP hooks, and pinned local bytes.

Hub itself is never mocked. Only HTTP transports and urllib responses supply
known own-model/data bytes. Every socket/DNS and NN forward is forbidden.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

from scripts.omni_revision import cpu_final_execution as runner

ROOT=Path('/workspace/work/omni-models-v1')
BASE=Path('/workspace/work/omni-cpu-final-execution-review-v1')
EVIDENCE=ROOT/'docs/course-revision-20261029-omni/infrastructure/cpu-final-execution-runtime-evidence-v1'
HOOK=r'''
import atexit,hashlib,io,json,os,socket,sys,urllib.parse,urllib.request
from pathlib import Path
import httpx
import torch
assert not any(k=='huggingface_hub' or k.startswith('huggingface_hub.') for k in sys.modules)
cache=Path('/workspace/tiny-perceptron-vlm/outputs/omni-revision/baseline')
archive=Path('/workspace/tiny-perceptron-vlm/assets/training/selftrained-v2.tar.gz')
a=cache/'moe-joint';b=cache/'b'
a_rev='979cdfacc588ad0536f1c64fff96f264571cf054';b_rev='2c6a1062f3e66a602cf83eda6c703456ec18bca0'
a_prefix='https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/'+a_rev+'/selftrained/v2/moe-joint/'
b_prefix='https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/'+b_rev+'/lesson-products/v1-20261016/b/'
assert hashlib.sha256((a/'inference-manifest.json').read_bytes()).hexdigest()=='f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e'
assert hashlib.sha256((b/'artifact-manifest.json').read_bytes()).hexdigest()=='dd077fd7a0f11fa5d95bdcf76a73bf6e035819ab135fa3191ff94f4e3643a1a3'
requests=[];urllib_calls=[];socket_calls=[];forward_calls=[]

def forbidden_socket(*a,**k):socket_calls.append(True);raise AssertionError('network forbidden')
socket.socket.connect=forbidden_socket;socket.getaddrinfo=forbidden_socket

def forbidden_forward(*a,**k):forward_calls.append(True);raise AssertionError('NN forward forbidden')
torch.nn.Module._call_impl=forbidden_forward

def handle(request):
    from huggingface_hub import constants
    assert constants.HF_HUB_OFFLINE is False and constants.is_offline_mode() is False
    assert os.environ.get('HF_HUB_OFFLINE')=='0'
    assert 'authorization' not in request.headers
    url=str(request.url);assert url.startswith(a_prefix),(request.method,url)
    name=urllib.parse.unquote(url[len(a_prefix):]);assert Path(name).name==name
    raw=(a/name).read_bytes();sha=hashlib.sha256(raw).hexdigest()
    requests.append({'method':request.method,'filename':name,'anonymous':True,'Hub_offline':False})
    headers={'Content-Length':str(len(raw)),'ETag':'"'+sha+'"','X-Repo-Commit':a_rev}
    assert request.method in ('HEAD','GET')
    return httpx.Response(200,headers=headers,content=b'' if request.method=='HEAD' else raw)
init=httpx.Client.__init__
def client_init(self,*args,**kw):kw['transport']=httpx.MockTransport(handle);return init(self,*args,**kw)
httpx.Client.__init__=client_init

def urlopen(request,timeout):
    if isinstance(request,str):
        assert request.startswith('https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/') and request.endswith('/assets/training/selftrained-v2.tar.gz') and timeout==60
        urllib_calls.append({'kind':'data','anonymous':True});return archive.open('rb')
    assert not any(k.lower()=='authorization' for k,_ in request.header_items())
    assert request.full_url.startswith(b_prefix) and timeout==30
    name=urllib.parse.unquote(request.full_url[len(b_prefix):]);assert Path(name).name==name
    urllib_calls.append({'kind':'B','filename':name,'anonymous':True});return io.BytesIO((b/name).read_bytes())
urllib.request.urlopen=urlopen

def done():
    from huggingface_hub import constants
    role='download' if sys.argv[0].endswith('legacy_download.py') else 'preflight' if sys.argv[0].endswith('core_final.py') else 'controller'
    report={'role':role,'argv':sys.argv,'requests':requests,'urllib_calls':urllib_calls,'socket_calls':len(socket_calls),'model_forward_calls':len(forward_calls),'final_offline_environment':os.environ.get('HF_HUB_OFFLINE'),'final_Hub_offline':constants.HF_HUB_OFFLINE,'formal_local_imported':'scripts.omni_revision.formal_local' in sys.modules}
    Path(os.environ['OMNI_REVIEW_RECEIPTS'],str(os.getpid())+'.json').write_text(json.dumps(report,indent=2)+'\n')
atexit.register(done)
'''
files=runner.source_paths();assert len(files)==99
expected={name:runner.file_sha(ROOT/name) for name in files}
with tempfile.TemporaryDirectory(prefix='runtime-cold-HTTP-',dir=BASE/'probes') as tmp:
    fixture=Path(tmp)/'source';fixture.mkdir();hooks=Path(tmp)/'hooks';hooks.mkdir();receipts=Path(tmp)/'receipts';receipts.mkdir();hf_home=Path(tmp)/'empty-HF';hf_home.mkdir()
    for name in files:
        target=fixture/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    hook = 'import os\ntry:\n' + '\n'.join('    ' + line for line in HOOK.splitlines()) + '\nexcept BaseException:\n    os._exit(90)\n'
    compile(hook, 'review-sitecustomize', 'exec')
    (hooks/'sitecustomize.py').write_text(hook)
    env={k:v for k,v in os.environ.items() if k not in {'GH_TOKEN','GITHUB_TOKEN','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE'}}
    env.update(PYTHONPATH=str(hooks)+os.pathsep+str(fixture),HF_HOME=str(hf_home),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_TOKEN='independent-test-placeholder',OMNI_REVIEW_RECEIPTS=str(receipts))
    code="from pathlib import Path; from scripts.omni_revision import cpu_final_execution as r; r.legacy_inputs(Path('control'))"
    process=subprocess.run([sys.executable,'-c',code],cwd=fixture,env=env,capture_output=True,text=True,timeout=60)
    if process.returncode:
        print(process.stdout);print(process.stderr);raise AssertionError('actual cold preparation failed')
    reports=[json.loads(p.read_text()) for p in receipts.glob('*.json')]
    assert sorted(x['role'] for x in reports)==['controller','download','preflight']
    down=next(x for x in reports if x['role']=='download');pre=next(x for x in reports if x['role']=='preflight');parent=next(x for x in reports if x['role']=='controller')
    assert len(down['requests'])==8 and sum(x['method']=='HEAD' for x in down['requests'])==4 and sum(x['method']=='GET' for x in down['requests'])==4
    assert sum(x['kind']=='B' for x in down['urllib_calls'])==6 and sum(x['kind']=='data' for x in down['urllib_calls'])==1
    assert down['formal_local_imported'] is False and down['final_Hub_offline'] is False and down['final_offline_environment']=='0'
    assert pre['formal_local_imported'] is True and pre['final_Hub_offline'] is True and pre['final_offline_environment']=='1' and pre['requests']==[] and pre['urllib_calls']==[]
    assert parent['final_Hub_offline'] is True and parent['final_offline_environment']=='1'
    assert all(x['socket_calls']==0 and x['model_forward_calls']==0 for x in reports)
    receipt=json.loads((fixture/'control/legacy-preflight.json').read_text());download=json.loads((fixture/'control/legacy-download.json').read_text())
    assert receipt['data_files']==8962 and receipt['data_bytes']==127161811 and receipt['no_model_predictions'] is True and receipt['old_B_depth']==1
    assert download['anonymous_A_download']['authentication']=='disabled'
    result={'reviewer_task':'/root/omni_cpu_final_execution_review','status':'pass_for_cold_fresh_process_preparation_with_real_Hub_HTTP_policy_and_mocked_HTTP','real_Controller_bounded_child_calls':True,'fresh_controller_download_preflight_processes':3,'empty_HF_cache':True,'inherited_HF_HUB_OFFLINE':'1','real_Hub_hf_hub_download_not_mocked':True,'real_Hub_request_event_hook_not_mocked':True,'HTTP_boundary':'httpx.MockTransport and exact pinned urllib fixture bytes; all socket/DNS forbidden','real_pinned_own_A_B_payloads_and_true_data_archive':True,'model_forward_calls':0,'network_socket_calls':0,'reports':reports,'download_receipt':download,'preflight_receipt':receipt,'files':expected}
    target=EVIDENCE/'cold-http-current-results.json';assert not target.exists();target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'pass','source_count':len(files),'real_Hub_HEAD':4,'real_Hub_GET':4,'anonymous_B_requests':6,'data_files':8962,'data_bytes':127161811,'model_forward_calls':0,'socket_calls':0,'offline_preflight':True,'parent_offline_preserved':True}))
