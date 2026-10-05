"""Fresh 14.1 bounded CPU checks; no training or existing weights are loaded."""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import random
import sys
import time

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT / 'docs/technical-reviews/artifacts/phase4-14_1-independent'
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.data import ByteTokenizer, IGNORE
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.modern import rope
from scripts.course_experiments.common import split_records, records_sha256, text_examples

torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None and not torch.cuda.is_available()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name, value):
    (BASE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

environment = {
    'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
    'torch_git_version': str(torch.version.git_version), 'device': 'cpu',
    'cuda_build': str(torch.version.cuda), 'cuda_available': str(torch.cuda.is_available()),
    'cwd': str(Path.cwd()), 'threads': str(torch.get_num_threads()),
    'offline_settings': {k: os.environ.get(k, '') for k in [
        'CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE']},
}
save('environment.json', environment)
raw = (BASE / 'section.md').read_bytes()
lines = raw.splitlines(keepends=True)
opened = False
code_lines = []
for line in lines:
    if not opened and line.startswith(b'```python'):
        opened = True
    elif opened and line.startswith(b'```'):
        break
    elif opened:
        code_lines.append(line)
code = b''.join(code_lines)
(BASE / 'original-fence.py').write_bytes(code)
namespace = {'__name__': '__main__'}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(code, 'course/chapters/14.md#14.1:original-fence', 'exec'), namespace)
printed = captured.getvalue()
assert printed.splitlines() == ['-0.7071', '-0.7071', '0.0'], repr(printed)
print('Original fence stdout:\n' + printed, end='')
score, rotate = namespace['score'], namespace['rotate']
exercise = {'score(22,25)': score(22,25), 'score(22,26)': score(22,26)}
assert round(exercise['score(22,25)'],4) == -0.7071
assert round(exercise['score(22,26)'],4) == -1.0
print('Exercise:', exercise)
angles = {}
for deg in [45,90,135]:
    z = rotate(torch.tensor([1.0,0.0]),deg/45)
    expected = torch.tensor([math.cos(math.radians(deg)),math.sin(math.radians(deg))])
    assert torch.allclose(z,expected,atol=1e-7,rtol=0)
    angles[str(deg)] = z.tolist()
print('Rotation components:', angles)
q, k = torch.tensor([-2.0,0.75]), torch.tensor([0.3,1.4])
base_dot = torch.dot(rotate(q,2),rotate(k,5)).item()
errors = []
for shift in [-4,1,10,20]:
    observed = torch.dot(rotate(q,2+shift),rotate(k,5+shift)).item()
    errors.append(abs(observed-base_dot))
    assert abs(observed-base_dot) <= 2e-6
    assert torch.allclose(rotate(q,2+shift).norm(),q.norm(),atol=5e-7,rtol=0)
print('Arbitrary content shared translation max abs dot error:',max(errors))
torch.manual_seed(7)
q_many,k_many = torch.randn(1,2,2,16),torch.randn(1,2,2,16)
pos = torch.tensor([2,5])
rq,rk = rope(q_many,pos),rope(k_many,pos)
shiftq,shiftk = rope(q_many,pos+10),rope(k_many,pos+10)
multi_error = (rq @ rk.transpose(-2,-1) - shiftq @ shiftk.transpose(-2,-1)).abs().max().item()
assert multi_error < 1e-5
assert torch.allclose(rq.norm(dim=-1),q_many.norm(dim=-1),atol=1e-6,rtol=0)
print('Original multi-pair rope [B,H,T,D]=[1,2,2,16], default frequencies:',
      (10000 ** (-torch.arange(0,16,2,dtype=torch.float32)/16)).tolist())
print('Multi-pair shared translation max abs dot error:',multi_error)
weights = torch.tensor([-0.7071,0.0]).softmax(-1)
assert bool((weights>=0).all()) and abs(weights.sum().item()-1)<1e-7
print('Negative scores become nonnegative softmax weights:',weights.tolist())
assert namespace['q'].grad is None and namespace['k'].grad is None

modern = ROOT / 'docs/course-experiments/results/modern.json'
result = json.loads(modern.read_text())
save('raw-json-top-schema.json',{k:type(v).__name__ for k,v in result.items()})
# Selected raw measurements only. No results/comparison, limitations, notes, or author summaries.
pointers = {}
def inspect(pointer):
    x = result
    for part in pointer.strip('/').split('/'):
        x = x[int(part)] if isinstance(x,list) else x[part]
    pointers[pointer] = x
    return x
for field in ['revision','seed','device','torch_version','python_version','gpu','step_scale']:
    inspect('/'+field)
code_files = ['tiny_perceptron/modern.py','tiny_perceptron/model.py','tiny_perceptron/attention.py',
              'tiny_perceptron/data.py','scripts/course_experiments/common.py']
for fn in code_files:
    # JSON pointer escape is recorded; direct lookup uses the original path key.
    p='/code_sha256/'+fn.replace('~','~0').replace('/','~1')
    pointers[p]=result['code_sha256'][fn]
    assert sha(ROOT/fn)==result['code_sha256'][fn],fn
historical=BASE/'inputs/architecture-at-run.py'
assert sha(historical)==result['code_sha256']['scripts/course_experiments/architecture.py']
pointers['/code_sha256/scripts~1course_experiments~1architecture.py']=sha(historical)
print('Experiment implementation hashes verified, historical architecture SHA:',sha(historical))
dataset_path = ROOT / 'data/training/text-initial/tinystories-train-512.jsonl'
records=[json.loads(line) for line in dataset_path.read_text().splitlines()]
assert len(records)==512
for record in records:
    record['family']=record.get('text_sha256',records_sha256([{'text':record['text']}]))
data=split_records(records,inspect('/seed'))
denominators={}
for split,rows in data.items():
    assert len(rows)==inspect('/results/dataset/'+split+'/records')
    assert records_sha256(rows)==inspect('/results/dataset/'+split+'/sha256')
    examples=text_examples(rows,'text',128)
    count=sum(int((y!=IGNORE).sum()) for _,y in examples)
    byte_eos_count=sum(len(r['text'].encode('utf-8'))+1 for r in rows)
    assert count==byte_eos_count
    denominators[split]={'records':len(rows),'examples':len(examples),'byte_plus_eos_targets':count,
                         'sha256':records_sha256(rows),'longest_input':max(len(x) for x,_ in examples)}
families=[{r['family'] for r in rows} for rows in data.values()]
assert all(not families[a]&families[b] for a in range(3) for b in range(a+1,3))
print('Reconstructed original split hashes and denominators:',json.dumps(denominators))
sampler=random.Random(result['seed'])
train_examples=text_examples(data['train'],'text',128)
sampled_targets=sum(sum(int((y!=IGNORE).sum()) for _,y in sampler.choices(train_examples,k=16))
                    for _ in range(240))
print('Replayed sampling only, not training: 240 batches x16; effective targets:',sampled_targets)
tok=ByteTokenizer()
print('UTF-8 raw bytes:',list('A'.encode('utf-8')),list('貓'.encode('utf-8')),
      'token ids:',tok.encode('A'),tok.encode('貓'))
assert len(tok.encode('A'))==1 and len(tok.encode('貓'))==3
models={}
for name in ['baseline','rope']:
    pref='/results/variants/'+name
    config=inspect(pref+'/model/config')
    model=TinyLM(ModelConfig(**config))
    parameters=sum(p.numel() for p in model.parameters())
    assert parameters==inspect(pref+'/model/parameters')
    for field in ['requested_steps','steps','optimizer_updates','skipped_updates','effective_tokens']:
        value=inspect(pref+'/training/'+field)
        assert value==({'requested_steps':240,'steps':240,'optimizer_updates':240,
                        'skipped_updates':0,'effective_tokens':sampled_targets}[field])
    with torch.no_grad():
        out=model(torch.tensor([[1,73,74,75,2]]))['logits']
    assert tuple(out.shape)==(1,5,264) and bool(torch.isfinite(out).all())
    try:
        model(torch.ones(1,129,dtype=torch.long))
    except ValueError as error:
        assert 'max_length' in str(error)
    else:
        raise AssertionError('Max length contract did not reject 129 positions')
    models[name]=parameters
    for split in ['validation','test']:
        p=pref+'/heldout/'+split
        nll=inspect(p+'/nll'); total=inspect(p+'/nll_sum'); count=inspect(p+'/effective_tokens')
        assert count==denominators[split]['byte_plus_eos_targets']
        assert inspect(p+'/records')==denominators[split]['records']
        assert inspect(p+'/examples')==denominators[split]['examples']
        assert abs(nll-total/count)<1e-12
        print(name,split,'nll',nll,'rounded5',round(nll,5),'sum',total,'denominator',count)
    sample=inspect(pref+'/heldout/test/samples/0')
    assert tok.decode(sample['generated_ids'])==sample['generated']
    assert len(sample['generated_ids'])==32
    print(name,'test sample0:',json.dumps(sample,ensure_ascii=False))
assert models['baseline']-models['rope']==128*64==8192
assert pointers['/results/variants/rope/heldout/test/samples/0']['prompt']=='Once upon a time, there '
assert pointers['/results/variants/rope/heldout/test/samples/0']['generated']=='was a loked there was a bough a '
print('Parameter counts:',models,'difference=128*64=8192')
save('inspected-raw-pointers.json',pointers)
save('verification-summary.json',{'original_fence_stdout':printed,'exercise':exercise,'components':angles,
    'arbitrary_content_shift_max_error':max(errors),'multi_pair_shift_max_error':multi_error,
    'denominators':denominators,'sampled_training_targets':sampled_targets,'parameters':models,
    'measurement_json_sha256':sha(modern),'original_data_sha256':sha(dataset_path),
    'scope':'Bounded CPU execution of original fence, variations, source helpers and data counts only. '
            'No optimizer step, saved model loading, long recipe, model scoring or GPU training rerun.'})
print('PASS: all bounded CPU assertions completed; no training or saved weights used.')
