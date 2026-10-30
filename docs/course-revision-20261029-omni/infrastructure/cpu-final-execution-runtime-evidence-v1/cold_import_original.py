"""Original cold-import proof: empty HF cache; network and NN calls forbidden."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path('/workspace/work/omni-models-v1')
PROBE=r'''
import json,os,socket,sys,tempfile
from unittest.mock import patch
initial=os.environ.get('HF_HUB_OFFLINE')
assert not any(k=='huggingface_hub' or k.startswith('huggingface_hub.') for k in sys.modules)
connections=[]
def forbidden(*args,**kw):
    connections.append('attempt');raise AssertionError('network forbidden')
with patch.object(socket.socket,'connect',forbidden),patch.object(socket,'getaddrinfo',forbidden):
    import torch
    with patch.object(torch.nn.Module,'_call_impl',side_effect=AssertionError('NN forbidden')):
        from scripts.omni_revision import cpu_final_execution
        import huggingface_hub.constants as constants
        from huggingface_hub import hf_hub_download
        after_import={'environment':os.environ.get('HF_HUB_OFFLINE'),'hub_constant':constants.HF_HUB_OFFLINE}
        os.environ['HF_HUB_OFFLINE']='0'
        after_late_override={'environment':os.environ['HF_HUB_OFFLINE'],'hub_constant':constants.HF_HUB_OFFLINE}
        with tempfile.TemporaryDirectory() as cache:
            try:hf_hub_download(repo_id='birdhackor/tiny-perceptron-course-models',revision='979cdfacc588ad0536f1c64fff96f264571cf054',filename='selftrained/v2/moe-joint/inference-manifest.json',token=False,cache_dir=cache)
            except Exception as error:
                chain=[];seen=set()
                while error is not None and id(error) not in seen:
                    seen.add(id(error));chain.append(type(error).__name__);error=error.__cause__ or error.__context__
            else:raise AssertionError('empty-cache offline original unexpectedly succeeded')
assert connections==[] and after_import=={'environment':'1','hub_constant':True} and after_late_override=={'environment':'0','hub_constant':True}
assert 'OfflineModeIsEnabled' in chain
print(json.dumps({'initial_offline_environment':initial,'after_controller_import':after_import,'after_late_environment_override':after_late_override,'real_Hub_download_error_chain':chain,'network_calls':0,'model_forward_calls':0}))
'''
results=[]
for setting in [None,'0']:
    with tempfile.TemporaryDirectory(prefix='cold-HF-') as cache:
        env={k:v for k,v in os.environ.items() if k not in {'HF_HUB_OFFLINE','GH_TOKEN','GITHUB_TOKEN','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET'}}
        env['HF_HOME']=cache;env['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
        if setting is not None:env['HF_HUB_OFFLINE']=setting
        value=subprocess.run([sys.executable,'-c',PROBE],cwd=ROOT,env=env,capture_output=True,text=True,check=True,timeout=30)
        results.append(json.loads(value.stdout))
path=ROOT/'docs/course-revision-20261029-omni/infrastructure/cpu-final-execution-runtime-evidence-v1/cold-import-original-results.json'
assert not path.exists();path.write_text(json.dumps({'reviewer_task':'/root/omni_cpu_final_execution_review','fresh_processes':2,'empty_cache':True,'results':results},indent=2)+'\n')
print(json.dumps({'fresh_processes':2,'network_calls':0,'model_forward_calls':0,'results':results},indent=2))
