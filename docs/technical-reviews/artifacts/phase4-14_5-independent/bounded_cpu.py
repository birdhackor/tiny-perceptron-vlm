"""Bounded CPU arithmetic, shape and existing-measurement audit. No training/evaluation."""
from pathlib import Path
import hashlib
import json
import math
import platform
import random
import sys

ROOT = Path('/workspace/tiny-perceptron-vlm')
PROOF = ROOT / 'docs/technical-reviews/artifacts/phase4-14_5-independent'
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from tiny_perceptron.modern import DenseFFN
from tiny_perceptron.model import ModelConfig, TinyLM

torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None
environment = {'python': platform.python_version(), 'python_executable': sys.executable, 'torch': str(torch.__version__), 'torch_git_version': str(torch.version.git_version), 'device': 'cpu', 'cuda_available': str(torch.cuda.is_available()), 'cuda_build': str(torch.version.cuda)}
(PROOF / 'bounded_cpu.environment.json').write_text(json.dumps(environment, indent=2) + '\n')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def dump(name, value):
    (PROOF / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

gate = torch.tensor([-2., 0., 2.], dtype=torch.float64)
content = torch.tensor([2., 2., 2.], dtype=torch.float64)
sigmoid = [1/(1+math.exp(-x)) for x in gate.tolist()]
expected = [x*s for x,s in zip(gate.tolist(), sigmoid)]
factor = F.silu(gate)
assert torch.allclose(factor, torch.tensor(expected, dtype=torch.float64), atol=1e-15, rtol=0)
assert content.shape == factor.shape == (3,)
assert (factor[0] < 0) and (factor[1] == 0) and (factor[2] > 1)
zero = F.silu(torch.zeros_like(gate)) * content
first_only = F.silu(torch.tensor([2., 0., 0.], dtype=torch.float64)) * content
assert zero.tolist() == [0., 0., 0.]
assert torch.allclose(first_only, torch.tensor([2*expected[2], 0., 0.], dtype=torch.float64), atol=1e-15, rtol=0)

ordinary = torch.nn.Sequential(torch.nn.Linear(8,16,bias=False), torch.nn.GELU(), torch.nn.Linear(16,8,bias=False))
three = torch.nn.ModuleList([torch.nn.Linear(8,16,bias=False),torch.nn.Linear(8,16,bias=False),torch.nn.Linear(16,8,bias=False)])
ordinary_count=sum(p.numel() for p in ordinary.parameters())
gated_count=sum(p.numel() for p in three.parameters())
assert (ordinary_count,gated_count)==(256,384)
assert 2*24*48 == 3*24*32 == 2304
small=DenseFFN(8,16,'swiglu').double()
x=torch.arange(48,dtype=torch.float64).reshape(2,3,8)/100
with torch.no_grad():
    h=small.up(x)
    g=small.gate(x)
    output=small(x)
    independent=F.linear(F.silu(F.linear(x,small.gate.weight,small.gate.bias))*F.linear(x,small.up.weight,small.up.bias),small.down.weight,small.down.bias)
assert h.shape==g.shape==(2,3,16)
assert output.shape==(2,3,8)
assert torch.equal(output,independent)
toy={'sigmoid_math':sigmoid,'silu':factor.tolist(),'modulated_content':(content*factor).tolist(),'zero_gate':zero.tolist(),'first_only_gate':first_only.tolist(),'bias_free_counts':{'ordinary':ordinary_count,'gated':gated_count},'matched_width_example':{'d':24,'ordinary_h':48,'swiglu_h':32,'both_weights':2304},'dense_ffn_shapes':{'input':list(x.shape),'content':list(h.shape),'gate':list(g.shape),'output':list(output.shape)},'tolerances':{'float64_silu_absolute':1e-15,'four_decimal_rounding':0.00005},'training_performed':False,'full_model_forward_performed':False}
dump('toy-results.json',toy)
print('TOY',json.dumps(toy,ensure_ascii=False))

original=PROOF/'inputs/original-modern.json'
raw=original.read_bytes(); source=json.loads(raw)
pointers=[]
def read(pointer):
    value=source
    for key in pointer.lstrip('/').split('/'):
        value=value[int(key)] if isinstance(value,list) else value[key]
    pointers.append(pointer)
    return value

selected={'original_result_sha256':sha(raw),'provenance':{},'variants':{},'pointers':pointers}
for name in ('revision','device','seed','torch_version','python_version','gpu','step_scale'):
    selected['provenance'][name]=read('/'+name)
selected['provenance']['runtime']=read('/results/runtime')
selected['provenance']['dataset']=read('/results/dataset')
asset=read('/assets/0')
selected['provenance']['asset']=asset
original_architecture=PROOF/'inputs/architecture-at-recorded-revision.py'
# JSON Pointer escaping for dictionary keys containing '/'.
for path in ('tiny_perceptron/modern.py','tiny_perceptron/model.py','tiny_perceptron/data.py','tiny_perceptron/attention.py','scripts/course_experiments/common.py','scripts/course_experiments/architecture.py'):
    expected_hash=source['code_sha256'][path]
    pointers.append('/code_sha256/'+path.replace('~','~0').replace('/','~1'))
    actual=sha(original_architecture.read_bytes()) if path.endswith('architecture.py') else sha((ROOT/path).read_bytes())
    assert expected_hash==actual, (path,expected_hash,actual)
    selected['provenance'].setdefault('verified_code_sha256',{})[path]=actual

for name in ('baseline','swiglu'):
    prefix='/results/variants/'+name
    model=read(prefix+'/model'); config=model['config']
    # Parameter enumeration of new CPU instances only; no trained weights, forward, gradients or evaluation.
    instance=TinyLM(ModelConfig(**config))
    parameters=sum(p.numel() for p in instance.parameters())
    bytes_=sum(p.numel()*p.element_size() for p in instance.parameters())
    assert parameters==model['parameters'] and bytes_==model['parameter_bytes']
    hidden={block.ffn.up.out_features for block in instance.blocks}
    assert hidden=={256}
    training={key:read(prefix+'/training/'+key) for key in ('requested_steps','steps','optimizer_updates','skipped_updates','batch_size','micro_batch_sizes','learning_rate','schedule','effective_tokens','warm_step_median_seconds')}
    assert training['requested_steps']==training['steps']==training['optimizer_updates']==240 and training['skipped_updates']==0
    heldout={}
    for split in ('validation','test'):
        measurements={key:read(prefix+'/heldout/'+split+'/'+key) for key in ('nll','nll_sum','effective_tokens','examples','records')}
        recomputed=measurements['nll_sum']/measurements['effective_tokens']
        assert abs(recomputed-measurements['nll'])<1e-12
        measurements['independent_nll_from_sum_and_tokens']=recomputed
        measurements['five_decimal_nll']=round(recomputed,5)
        heldout[split]=measurements
    selected['variants'][name]={'model':model,'parameter_count_recomputed_on_cpu':parameters,'hidden_widths':sorted(hidden),'training':training,'heldout':heldout,'milliseconds_from_seconds':1000*training['warm_step_median_seconds']}

assert selected['variants']['swiglu']['model']['parameters']-selected['variants']['baseline']['model']['parameters']==2*(64*256+256)==33280
assert round(selected['variants']['baseline']['milliseconds_from_seconds'],3)==13.062
assert round(selected['variants']['swiglu']['milliseconds_from_seconds'],3)==13.336
assert [selected['variants']['baseline']['heldout'][s]['five_decimal_nll'] for s in ('validation','test')]==[2.26331,2.29163]
assert [selected['variants']['swiglu']['heldout'][s]['five_decimal_nll'] for s in ('validation','test')]==[2.25444,2.28009]

dataset_bytes=(PROOF/'inputs/original-tinystories-train-512.jsonl').read_bytes()
member=next(f for f in asset['files'] if f['path'].endswith('/tinystories-train-512.jsonl'))
assert sha(dataset_bytes)==member['sha256']
rows=[json.loads(line) for line in dataset_bytes.splitlines()]
expected_keys={'id','text','split','source_split','source_record_index','source','source_revision','language','license','text_sha256'}
assert all(set(row)<=expected_keys for row in rows)
for row in rows:
    row['family']=row.get('text_sha256',sha(json.dumps([{'text':row['text']}],ensure_ascii=False,sort_keys=True).encode()))
groups={};seen=set()
for row in rows:
    encoded=json.dumps(row,ensure_ascii=False,sort_keys=True)
    if encoded in seen: continue
    seen.add(encoded)
    key=str(row.get('family',sha(encoded.encode())))
    groups.setdefault(key,[]).append(row)
keys=sorted(groups);random.Random(source['seed']).shuffle(keys)
a=min(max(1,int(len(keys)*.8)),len(keys)-2)
b=min(max(a+1,int(len(keys)*.9)),len(keys)-1)
splits={name:[row for key in chosen for row in groups[key]] for name,chosen in (('train',keys[:a]),('validation',keys[a:b]),('test',keys[b:]))}
serialized=(json.dumps(splits,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
artifact=next(a for a in read('/artifacts') if a['path']=='dataset.json')
assert sha(serialized)==artifact['sha256'] and len(serialized)==artifact['bytes']
counts={}
for split,records in splits.items():
    digest=sha(json.dumps(records,sort_keys=True,ensure_ascii=False).encode())
    assert digest==source['results']['dataset'][split]['sha256']
    assert len(records)==source['results']['dataset'][split]['records']
    lengths=[]
    for row in records:
        # BOS + UTF8 bytes + EOS -> next-token targets: bytes + EOS, each target once.
        targets=len(row['text'].encode('utf-8'))+1
        lengths.extend(min(128,targets-start) for start in range(0,targets,128))
    counts[split]={'records':len(records),'records_sha256':digest,'effective_tokens':sum(lengths),'examples':len(lengths)}
    if split=='train':
        sampler=random.Random(source['seed']);drawn=sum(sum(sampler.choices(lengths,k=16)) for _ in range(240))
        counts[split]['sampled_training_tokens_240x16']=drawn
        assert drawn==452102
    else:
        for name in ('baseline','swiglu'):
            heldout=selected['variants'][name]['heldout'][split]
            assert all(heldout[k]==counts[split][k] for k in ('records','effective_tokens','examples'))
selected['independent_data_denominators']=counts
selected['reconstructed_original_dataset_artifact']={'sha256':sha(serialized),'bytes':len(serialized),'matches_original_artifact':True,'training_data_downloaded':False}
selected['latency_scope']='Original GPU raw median field converted from seconds to milliseconds; original code computes median after excluding first 3 updates and synchronizes before/after each timed step. Raw per-step latency vector was not exported, no GPU timing repeated.'
selected['audit_scope']='Existing-measurement arithmetic and integer sampling reconstruction only; no model training, GPU work, trained-weight loading or full model evaluation.'
dump('raw-measurement-audit.json',selected)
print('EMPIRICAL',json.dumps({'original_result_sha256':selected['original_result_sha256'],'parameters':{k:v['parameter_count_recomputed_on_cpu'] for k,v in selected['variants'].items()},'denominators':counts,'dataset_artifact_sha256':sha(serialized),'nll':{k:{s:v['heldout'][s]['five_decimal_nll'] for s in ('validation','test')} for k,v in selected['variants'].items()},'milliseconds':{k:round(v['milliseconds_from_seconds'],3) for k,v in selected['variants'].items()},'training_performed':False,'full_model_evaluation_performed':False},ensure_ascii=False))
