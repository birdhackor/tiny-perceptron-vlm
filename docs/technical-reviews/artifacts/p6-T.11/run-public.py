"""Two selected public CPU interface checks, not heldout re-evaluation."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
commands=[]
for task,example in [('text','text.messages.json'),('tool_call','tool_call.messages.json')]:
    command=[sys.executable,'scripts/selftrained/chat.py','--model-dir','outputs/p6-T.11-public/moe-joint',
        '--asset-dir','outputs/selftrained-v2/data','--repo','birdhackor/tiny-perceptron-course-models',
        '--revision','979cdfacc588ad0536f1c64fff96f264571cf054','--prefix','selftrained/v2/moe-joint',
        '--manifest-sha256','f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e',
        '--messages','docs/selftrained/examples/v2/'+example,'--task',task,'--device','cpu','--max-new-tokens','128','--threads','2']
    if task=='tool_call':command.append('--tools')
    t=time.monotonic();run=subprocess.run(command,cwd=ROOT,capture_output=True,timeout=90)
    (OUT/(task+'.actual.stdout.json')).write_bytes(run.stdout)
    (OUT/(task+'.actual.stderr.txt')).write_bytes(run.stderr)
    assert run.returncode==0,run.stderr.decode()
    response=json.loads(run.stdout)
    original=json.loads((ROOT/f'docs/selftrained/results/public-cpu-raw/{task}-1-result.json').read_bytes())
    assert response['answer']==original['actual_answer']
    if task=='tool_call':
        assert response['tool_trace']['executed'] and len(response['generations'])==2
        assert response['generations'][1]['prompt_messages'][-1]['role']=='tool'
    commands.append({'task':task,'argv':command,'returncode':run.returncode,'elapsed_seconds':time.monotonic()-t,
        'answer':response['answer'],'generation_count':len(response['generations']),
        'stops':[g['generation_status'] for g in response['generations']],
        'stdout_sha256':hashlib.sha256(run.stdout).hexdigest(),'matches_archived_answer':True,
        'public_source':response['public_source'],'selected_step':response['model']['selected_step'],
        'files':response['model']['files']})
result={'commands':commands,'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'},
        'scope':'two published validation examples, no heldout data or general-quality estimate'}
(OUT/'public-actual-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'environment':result['environment'],'tasks':[{'task':c['task'],'answer':c['answer'],'generations':c['generation_count'],'stops':c['stops']} for c in commands]},ensure_ascii=False,indent=2))
