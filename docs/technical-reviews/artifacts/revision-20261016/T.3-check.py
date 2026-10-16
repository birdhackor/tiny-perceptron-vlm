import contextlib
import hashlib
import io
import json
import os
import platform
import runpy
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path('/workspace/tiny-perceptron-vlm')
WORK = Path('/workspace/work/curriculum-20261016/T.3')
PY = ROOT / '.venv/bin/python'
sys.path.insert(0, str(ROOT))
from scripts.train_simple import examples
from tiny_perceptron.data import ByteTokenizer, split_documents, toy_documents
from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.tokenization import generation_report
from tiny_perceptron.model import generate

record = {'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'platform': platform.platform(), 'device': 'cpu', 'windows_runtime': 'not executed; platform syntax checked against official docs'}, 'commands': []}

def run(args, *, expected=0):
    result = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True, text=True, timeout=120)
    entry = {'command': shlex.join(str(x) for x in args), 'cwd': str(ROOT), 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    record['commands'].append(entry)
    if result.returncode != expected:
        raise RuntimeError(entry)
    return result

run(['mkdir', '-p', WORK / 'outputs', WORK / 'checkpoints'])
for model in ('bigram', 'mlp'):
    common = [PY, 'scripts/train_simple.py', '--model', model]
    if model == 'mlp': common += ['--context', '3', '--width', '16']
    common += ['--seed', '42', '--device', 'cpu']
    for train in (False, True):
        args = common + (['--train', '--steps', '200', '--output', WORK / 'checkpoints' / f'{model}.pt'] if train else [])
        result = run(args)
        phase = 'after' if train else 'before'
        (WORK / 'outputs' / f'{model}-{phase}.json').write_text(result.stdout)

torch.set_num_threads(2)
documents = toy_documents()
parts = split_documents(documents, 42)
vocabulary = {c: i + 2 for i, c in enumerate(sorted(set(''.join(parts['train']))))}
record['data'] = {'documents': len(documents), 'unique_documents': len(set(documents)), 'split_counts': {k: len(v) for k, v in parts.items()}, 'train_vocabulary_characters': len(vocabulary), 'total_simple_ids': len(vocabulary)+2, 'denominators': {k: {'characters': sum(len(x) for x in v), 'next_character_targets_including_end_boundary': sum(len(x)+1 for x in v)} for k,v in parts.items()}}

record['saved_checkpoint_evaluation'] = {}
for model_kind in ('bigram', 'mlp'):
    payload = torch.load(WORK / 'checkpoints' / f'{model_kind}.pt', weights_only=True)
    model = BigramLM(len(vocabulary)+2) if model_kind == 'bigram' else ContextMLP(len(vocabulary)+2,3,16)
    model.load_state_dict(payload['model'])
    after = json.loads((WORK / 'outputs' / f'{model_kind}-after.json').read_text())
    before = json.loads((WORK / 'outputs' / f'{model_kind}-before.json').read_text())
    results = {'checkpoint_keys': list(payload), 'parameter_shapes': {k: list(v.shape) for k,v in model.state_dict().items()}, 'parameters': sum(p.numel() for p in model.parameters()), 'last_before_update': after['last_batch_loss_before_update'], 'deltas_after_minus_before': {k: after[k]-before[k] for k in ('train_loss','validation_loss')}}
    with torch.no_grad():
        for split, report_name in [('train','train_loss'), ('validation','validation_loss')]:
            x,y=examples(parts[split],vocabulary,payload['context'])
            ce = F.cross_entropy(model(x),y).item()
            explicit_mean = (-F.log_softmax(model(x),-1)[torch.arange(len(y)),y]).mean().item()
            results[split] = {'targets': len(y), 'loss_from_saved_state': ce, 'explicit_mean_nll': explicit_mean, 'reported_loss': after[report_name], 'absolute_error': abs(ce-after[report_name])}
            assert abs(ce-after[report_name]) < 1e-6
            assert abs(ce-explicit_mean) < 1e-6
    record['saved_checkpoint_evaluation'][model_kind] = results
    run([PY, 'scripts/infer.py', WORK/'checkpoints'/f'{model_kind}.pt','--device','cpu','--json'],expected=1)

# Instrument the actual dry-run main: intercept optimizer construction, compare
# its parameter tensors before and after, and count calls to step().
record['dry_run_no_update']={}
original_optimizer = torch.optim.AdamW
for model_kind in ('bigram','mlp'):
    captured={}
    def optimizer(*args,**kwargs):
        opt=original_optimizer(*args,**kwargs)
        captured['optimizer']=opt
        captured['before']=[p.detach().clone() for group in opt.param_groups for p in group['params']]
        captured['step_count']=0
        original_step=opt.step
        def step(*a,**kw):
            captured['step_count']+=1
            return original_step(*a,**kw)
        opt.step=step
        return opt
    stdout=io.StringIO()
    with patch.object(sys,'argv',['scripts/train_simple.py','--model',model_kind,'--seed','42','--device','cpu']),patch.object(torch.optim,'AdamW',optimizer),contextlib.redirect_stdout(stdout):
        runpy.run_path(str(ROOT/'scripts/train_simple.py'),run_name='__main__')
    after=[p.detach() for group in captured['optimizer'].param_groups for p in group['params']]
    same=all(torch.equal(a,b) for a,b in zip(captured['before'],after))
    record['dry_run_no_update'][model_kind]={'step_calls':captured['step_count'],'parameters_bitwise_unchanged':same,'stdout':json.loads(stdout.getvalue())}
    assert captured['step_count']==0 and same

tok=ByteTokenizer()
record['byte_example']={'encode_A':tok.encode('A'),'eos_id':tok.eos_id,'report':generation_report(tok,[73,2])}
assert record['byte_example']['encode_A']==[73]
assert record['byte_example']['report']['answer']=='A'

class Scripted(torch.nn.Module):
    def __init__(self, values, max_length=128):
        super().__init__()
        self.values=values
        self.config=SimpleNamespace(max_length=max_length)
        self.calls=0
    def forward(self,x,cache=None):
        value=self.values[min(self.calls,len(self.values)-1)]
        self.calls+=1
        logits=torch.full((1,x.shape[1],264),-100.0)
        logits[0,-1,value]=100
        return {'logits':logits,'cache':None}
record['generation_stop']={}
for label,values,limit in [('eos',[73,2],128),('token_limit',[73],128),('context_limit',[73],3)]:
    model=Scripted(values,limit)
    out=generate(model,torch.tensor([[1]]),32,eos_id=2)
    result=generation_report(tok,out[0,1:].tolist())
    record['generation_stop'][label]={'new_ids':out.shape[1]-1,'model_calls':model.calls,'report':result}
assert record['generation_stop']['eos']['new_ids']==2
assert record['generation_stop']['token_limit']['new_ids']==32
assert record['generation_stop']['context_limit']['new_ids']==2

run([PY,'scripts/fetch_course_models.py','--list'])
run([PY,'scripts/fetch_course_models.py','--model','text_foundation','--output',WORK/'course-models'])
model_dir=WORK/'course-models/text_foundation'
run([PY,'scripts/infer.py',model_dir/'model.pt','--prompt','color=blue;shape=circle;','--tokens','32','--device','cpu','--json'])
payload=torch.load(model_dir/'model.pt',weights_only=True)
record['public_checkpoint']={'keys':list(payload),'has_optimizer':'optimizer' in payload,'has_training_state':'training_state' in payload,'format_version':payload.get('format_version'),'metadata':payload.get('metadata'),'config':payload.get('config')}
card=(model_dir/'README.md').read_text()
record['public_card']={'sha256':hashlib.sha256((model_dir/'README.md').read_bytes()).hexdigest(),'excerpts':[line for line in card.splitlines() if 'from-scratch' in line or 'optimizer' in line]}

# This is the local, finite CPU recipe only; no Modal/remote training.
run([PY,'-m','scripts.course_experiments.run','--experiment','simple_models','--device','cpu','--output',WORK/'course-experiments'])
r=json.loads((WORK/'course-experiments/simple_models/result.json').read_text())
record['fixed_comparison']={'experiment_id':r['experiment_id'],'seed':r['seed'],'device':r['device'],'step_scale':r['step_scale'],'evidence_status':r['evidence_status'],'runs':{k:{f:v[f] for f in ['parameters','context','steps','before_nll','after_nll_same_post_update_time']} for k,v in r['results']['runs'].items()},'data':r['results']['data']}
record['notes']=['Output paths were redirected to isolated work/T.3 equivalents to avoid touching shared outputs/checkpoints. Model parameters and 200 CPU updates are the exact lesson recipe.','No native Windows or macOS execution. No GPU or remote training. No page-layout inspection; T.3 has no figures.','The scripted generation model checks stopping mechanics, not model quality. Public-model answer is reported as observed without claiming correctness.']
artifact=ROOT/'docs/technical-reviews/artifacts/revision-20261016/T.3-execution.json'
artifact.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'artifact':str(artifact),'environment':record['environment'],'data':record['data'],'byte_example':record['byte_example'],'dry_run_no_update':record['dry_run_no_update'],'saved_checkpoint_evaluation':record['saved_checkpoint_evaluation'],'fixed_comparison':record['fixed_comparison']},ensure_ascii=False,indent=2))
