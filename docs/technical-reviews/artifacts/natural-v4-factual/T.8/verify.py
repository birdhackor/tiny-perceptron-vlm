"""Independent bounded T.8 CPU probes and original-record arithmetic audit."""
import copy
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
from scripts.prepare_data import generate_records
from scripts.course_experiments.common import Context, split_records, records_sha256, text_examples
from scripts.course_experiments import architecture as a
from scripts.course_experiments.run import execute, experiment_spec
from tiny_perceptron.data import ByteTokenizer, load_jsonl, IGNORE, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum
from tiny_perceptron.training import seed_everything, load_checkpoint

ROOT = Path(__file__).resolve().parents[5]
TMP = ROOT / 'outputs/natural-v4/factual-research/T.8'
OUT = Path(__file__).parent
torch.set_num_threads(2)
env = {'python':platform.python_version(), 'torch':str(torch.__version__),
       'torch_git_version':torch.version.git_version, 'device':'cpu', 'cuda_available':str(torch.cuda.is_available())}
print('OWN ENVIRONMENT',json.dumps(env))
assert str(torch.__version__) == '2.14.1+cpu' and not torch.cuda.is_available()
tok=ByteTokenizer()

def emit(name, value):
    print(name,json.dumps(value,ensure_ascii=False,sort_keys=True))

def targets(rows, mode='text'):
    return sum(int((y!=IGNORE).sum()) for _,y in text_examples(rows,mode))

def schedule(rows,steps,batch_size,mode='text'):
    examples=text_examples(rows,mode); rng=random.Random(42)
    return sum(sum(int((y!=IGNORE).sum()) for _,y in rng.choices(examples,k=batch_size)) for _ in range(steps))

original_path=ROOT/'data/training/text-initial/tinystories-train-512.jsonl'
rows=load_jsonl(original_path)
assert len(rows)==512 and hashlib.sha256(original_path.read_bytes()).hexdigest()=='7aa55a657de6499be64a513aa76eb21a9cd52276bf836ba725e1d7f52ea511e0'
for r in rows:
    assert r['text_sha256']==hashlib.sha256(r['text'].encode()).hexdigest()
    r['family']=r['text_sha256']
data=split_records(rows,42)
sft=split_records(generate_records('attributes-sft'),42)
emit('RECOMPUTED DATA',{'stories':{k:{'records':len(v),'sha256':records_sha256(v),'targets':targets(v)} for k,v in data.items()},
    'sft':{k:{'records':len(v),'families':len({r['family'] for r in v}),'sha256':records_sha256(v),'targets':targets(v,'sft')} for k,v in sft.items()},
    'modern_train_targets':schedule(data['train'],240,16),'moe_train_targets':schedule(data['train'],180,16),
    'efficiency_100_targets':schedule(sft['train'],100,16,'sft'),'efficiency_40_targets':schedule(sft['train'],40,8,'sft'),
    'precision_200_targets':schedule(sft['train'],200,16,'sft')})
assert [len(data[k]) for k in ['train','validation','test']]==[409,51,52]
assert [targets(data[k]) for k in ['validation','test']]==[39256,41914]
assert schedule(data['train'],240,16)==452102 and schedule(data['train'],180,16)==337761
assert [targets(sft[k],'sft') for k in ['validation','test']]==[36,69]

config=ModelConfig(width=64,layers=2,heads=4)
expected_counts={'baseline':141568,'rope':133376,'rmsnorm':141248,'relu2':141568,'swiglu':174848,'tied':124672}
changes={'baseline':{},'rope':{'rotary':True},'rmsnorm':{'norm':'rms'},'relu2':{'activation':'relu2'},'swiglu':{'activation':'swiglu'},'tied':{'tied':True}}
base=TinyLM(config)
ctx=Context('cpu',TMP/'probe',TMP,ROOT/'assets/training',seed=42);ctx.output.mkdir(exist_ok=True)
for name,change in changes.items():
    m,copied=a._clone_config(base,ctx,**change)
    count=sum(p.numel() for p in m.parameters());assert count==expected_counts[name]
    assert all(torch.equal(m.state_dict()[n],base.state_dict()[n]) for n in copied)
    if name=='tied': assert m.output.weight is m.embedding.weight and torch.equal(m.output.weight,base.embedding.weight)
    emit('OWN PARAMETER COUNT',{'variant':name,'parameters':count})
moe=TinyLM(replace(config,experts=4,top_k=2));budget=a._parameter_budget(moe)
assert budget=={'total_parameters':340608,'active_parameter_proxy_per_token':208256,'expert_parameters':264704,'router_parameters':512}
emit('OWN MOE BUDGET',budget)
for target in [142080,208256,340608]: emit('OWN MATCHED DENSE',{'target':target,'width_and_actual_minus_target':a._matched_width(config,target)})
assert a._matched_width(config,208256)==(80,-576) and a._matched_width(config,340608)==(104,-10720)
assert 141568-128*64==133376 and 141568-5*64==141248 and 141568-264*64==124672
assert 141568+2*(64*256+256)==174848

def load_record(name):
    p=ROOT/f'docs/course-experiments/results/{name}.json';d=json.loads(p.read_text())
    assert d['device']=='cuda' and d['gpu']=='NVIDIA L4' and d['torch_version']=='2.14.1+cu126' and d['seed']==42
    assert d['step_scale']==1 and d['evidence_status']=='complete_run'
    emit('ORIGINAL RECORD',{'name':name,'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'revision':d['revision']})
    return d

expected_modern=[(2.26331,2.29163,13.062),(1.74158,1.77812,14.905),(2.25913,2.28631,13.952),(2.23590,2.26438,12.893),(2.25444,2.28009,13.336),(2.36276,2.38422,12.768)]
for record_name in ['modern','moe']:
    d=load_record(record_name);r=d['results'];limit=240 if record_name=='modern' else 180
    assert r['dataset']=={k:{'records':len(v),'sha256':records_sha256(v)} for k,v in data.items()}
    for i,(name,v) in enumerate(r['variants'].items()):
        t=v['training'];assert t['steps']==t['optimizer_updates']==limit and t['skipped_updates']==0
        assert t['effective_tokens']==schedule(data['train'],limit,16)
        c=v['model']['config'];m=TinyLM(ModelConfig(**c));assert sum(p.numel() for p in m.parameters())==v['model']['parameters']
        vals=[]
        for split in ['validation','test']:
            h=v['heldout'][split];assert h['records']==len(data[split]) and h['effective_tokens']==targets(data[split])
            assert abs(h['nll_sum']/h['effective_tokens']-h['nll'])<1e-12
            assert len(h['samples'])==8
            for row,sample in zip(data[split],h['samples']):
                assert sample['prompt']==row['text'][:24]
                assert tok.decode(sample['generated_ids'])==sample['generated']
            vals.append(round(h['nll'],5))
        ms=round(t['warm_step_median_seconds']*1000,3)
        if record_name=='modern':assert (*vals,ms)==expected_modern[i] and v['model']['parameters']==expected_counts[name]
        if record_name=='moe' and name in ['top2_aux0.01','dense_active_top2','dense_total']:
            expected={'top2_aux0.01':(2.30170,27.242),'dense_active_top2':(2.30906,11.965),'dense_total':(2.24943,14.054)}[name]
            assert (vals[1],ms)==expected
        if record_name=='moe' and v.get('validation_routing'):
            for layer in v['validation_routing']['layers']:
                assert sum(layer['dispatch_counts'])==layer['dispatch_denominator']==39256*c['top_k']
        emit('AUDITED TRAINING ROW',{'experiment':record_name,'variant':name,'parameters':v['model']['parameters'],'heldout_nll':vals,'median_update_ms':ms,'first_test_sample':v['heldout']['test']['samples'][0]})

def check_sft_heldout(h):
    for split in ['validation','test']:
        v=h[split]; assert v['effective_tokens']==targets(sft[split],'sft') and v['records']==len(sft[split])
        matches=0
        for row,sample in zip(sft[split],v['samples']):
            assert sample['expected']==row['messages'][-1]['content']
            ids=sample['generated_ids'];raw=ids[:ids.index(2)] if 2 in ids else ids
            exact=raw==tok.encode(sample['expected']); assert sample['exact']==exact
            matches+=exact
        assert matches==v['matches'] and v['exact_match']==matches/v['records']

eff=load_record('efficiency')['results'];precision=load_record('precision')['results']
for r in [eff,precision]:assert r['dataset']=={k:{'records':len(v),'sha256':records_sha256(v)} for k,v in sft.items()}
for name,v in list(eff['models'].items())+list(eff['update_variants'].items())+list(precision['variants'].items()):
    t=v['training'];check_sft_heldout(v['heldout'])
    steps=100 if name in ['mha','gqa'] else 200 if name in ['fp32','bf16','fp16'] else 40
    batch=8 if steps==40 else 16
    assert t['steps']==t['optimizer_updates']==steps and t['skipped_updates']==0
    assert t['effective_tokens']==schedule(sft['train'],steps,batch,'sft')
    before,peak,extra=[t[k] for k in ['memory_allocated_before_bytes','peak_memory_allocated_bytes','peak_additional_allocated_bytes']]
    assert peak-before==extra
    emit('AUDITED SFT ROW',{'variant':name,'steps':steps,'training_targets':t['effective_tokens'],'validation_test_matches':[v['heldout'][k]['matches'] for k in ['validation','test']],
        'median_update_ms':1000*t['warm_step_median_seconds'],'memory_MiB':[n/2**20 for n in [before,peak,extra]],'model_config':v.get('model',{}).get('config')})
for name in ['padded','packed']:
    t=eff['packing']['actual_updates'][name];assert t['steps']==40 and t['effective_tokens']==845
    assert all(k not in t for k in ['memory_allocated_before_bytes','peak_memory_allocated_bytes','peak_additional_allocated_bytes'])
    check_sft_heldout(t['heldout']);assert t['heldout']['test']['matches']==1
compile=eff['compile'];assert round(compile['first_compiled_call_seconds'],3)==7.717 and compile['backend']=='inductor' and compile['shape']==[4,16,64]
assert compile['estimated_calls_to_amortize_first_call'] is None
for key in ['eager','compiled']:
    assert compile[key]['median_seconds']==statistics.median(compile[key]['samples_seconds'])
emit('AUDITED EFFICIENCY MECHANISMS',{k:eff[k] for k in ['manual_vs_sdpa','accumulation','activation_checkpoint','compile']})
for v in eff['models'].values():
    c=v['cache']; assert c['generated_ids_full']==c['generated_ids_cached'] and max(c['per_step_logit_max_error'])<1e-5
    for k in ['prefill','full_recompute_decode','cached_decode']:assert c[k]['median_seconds']==statistics.median(c[k]['samples_seconds']) and c[k]['warmup_calls']==3 and c[k]['measured_calls']==9
assert 'memory_efficient'==eff['manual_vs_sdpa']['backend']['selected']
assert [precision['variants'][k]['heldout']['test']['matches'] for k in ['fp32','bf16','fp16']]==[7,6,4]
for k,v in precision['variants'].items():
    assert v['weights_dtype']=='torch.float32' and v['training']['all_parameters_finite'] and v['logits_finite']
    assert v['model']['parameter_bytes']==566272
    assert v['observed_logits_dtype']=={'fp32':'torch.float32','bf16':'torch.bfloat16','fp16':'torch.float16'}[k]
    assert len(v['inference']['samples_seconds'])==9 and v['inference']['warmup_calls']==3
emit('PRECISION OBSERVED DTYPES',{k:{'weights':v['weights_dtype'],'logits':v['observed_logits_dtype'],'scaler':v['training']['scaler_enabled'],'history':v['training']['history']} for k,v in precision['variants'].items()})
flash_record=load_record('flash_probe');f=flash_record['results']
assert f['configuration']['shape_B_H_T_D']==[2,4,512,32] and f['runtime']['compute_capability']==[8,9]
assert f['runtime']['driver_query']['stdout']=='NVIDIA L4, 580.95.05'
for name,r in f['routes'].items():
    assert r['status']=='completed'
    for check in [r['correctness']['output'],*r['correctness']['gradients'].values()]:assert check['finite'] and check['within_declared_tolerance']
    for profile in r['profiles'].values():assert profile['flash_cuda_verified'] and profile['cuda_kernel_names']
    for mode,route in r['measurements'].items():
        for backend,m in route.items():
            assert len(m['samples_seconds'])==9 and m['measured_calls']==9 and m['warmup_calls']==3 and m['synchronized']
            assert m['median_seconds']==statistics.median(m['samples_seconds'])
            assert m['peak_allocated_bytes']-m['allocated_before_bytes']==m['additional_peak_allocated_bytes']
            emit('AUDITED FLASH MEASUREMENT',{'dtype':name,'mode':mode,'backend':backend,'median_ms':round(1000*m['median_seconds'],3),
                'MiB':[m[k]/2**20 for k in ['allocated_before_bytes','peak_allocated_bytes','additional_peak_allocated_bytes']]})
    emit('AUDITED FLASH NUMERICS',{'dtype':name,'correctness':r['correctness'],'profiles':r['profiles']})
try:
    execute('flash_probe','cuda',TMP/'must-not-exist',TMP,ROOT/'assets/training')
    raise AssertionError('CUDA guard failed')
except RuntimeError as error:
    assert str(error)=='A GPU experiment must not silently fall back to CPU'
    emit('OWN CUDA GUARD',str(error))

# Bounded own software/gradient/AMP checks. These are not GPU replications.
small=TinyLM(ModelConfig(width=8,layers=2,heads=2,max_length=128))
probe=a._sdpa_probe(small,text_examples(sft['train'],'sft'),ctx)
assert probe['output_max_error']<1e-5 and probe['gradient_max_error']<1e-6
emit('OWN CPU SDPA',probe)
ck=a._checkpoint_probe(small,text_examples(sft['train'],'sft'),ctx);assert ck['logit_max_error']==ck['gradient_max_error']==0
emit('OWN CPU CHECKPOINT',ck)
acc=a._accumulation_probe(small,a._mechanism_examples(sft['train'],128),ctx);assert acc['shared_denominator']==20 and acc['gradient_max_error']<1e-6
emit('OWN CPU ACCUMULATION',acc)
m=copy.deepcopy(small);tr=a._train(m,sft['train'],ctx,name='cpu-bf16-two-step',steps=2,mode='sft',dtype=torch.bfloat16)
payload=torch.load(ctx.output/'cpu-bf16-two-step.pt',weights_only=True)
state_dtypes={str(v.dtype) for s in payload['optimizer']['state'].values() for v in s.values() if isinstance(v,torch.Tensor)}
assert {str(p.dtype) for p in m.parameters()}=={'torch.float32'} and state_dtypes=={'torch.float32'}
x,y,valid=pad_batch(text_examples(sft['validation'],'sft'))
with torch.autocast('cpu',dtype=torch.bfloat16):logits=m(x,valid=valid)['logits']
assert logits.dtype==torch.bfloat16 and torch.isfinite(logits).all()
emit('OWN CPU AMP',{'weights':'torch.float32','optimizer_state_dtypes':sorted(state_dtypes),'logits':str(logits.dtype),'updates':tr['optimizer_updates'],'skipped':tr['skipped_updates']})

# Execute the three actual small CLI recipes, changing only output locations and device.
commands=[[sys.executable,'scripts/prepare_data.py','--kind','toy-text','--seed','42','--output',str(TMP/'toy-data')]]
for name,extra in [('baseline',[]),('rms',['--norm','rms']),('moe',['--experts','4','--top-k','2'])]:
    commands.append([sys.executable,'scripts/train.py','--task','text','--data',str(TMP/'toy-data/toy-text/train.jsonl'),'--train','--steps','200','--seed','42','--device','cpu','--output',str(TMP/f'toy-{name}.pt'),*extra])
    commands.append([sys.executable,'scripts/evaluate.py',str(TMP/f'toy-{name}.pt'),'--data',str(TMP/'toy-data/toy-text/validation.jsonl'),'--mode','text','--tokens','32','--device','cpu','--output',str(TMP/f'toy-{name}-validation.json')])
for i,command in enumerate(commands):
    p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90)
    (TMP/f'cli-{i}.stdout.txt').write_text(p.stdout);(TMP/f'cli-{i}.stderr.txt').write_text(p.stderr)
    emit('OWN CLI',{'command':command,'native_exit':p.returncode,'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'stderr':p.stderr})
    assert p.returncode==0
for name in ['baseline','rms','moe']:
    e=json.loads((TMP/f'toy-{name}-validation.json').read_text());t=json.loads((TMP/f'toy-{name}.json').read_text())
    assert e['effective_tokens']==35 and len(e['samples'])==1 and e['samples'][0]['row']==0 and e['samples'][0]['prompt']=='colo' and e['skipped']==[]
    assert e['mean_token_nll']>0 and t['parameters']>0 and t['seconds']>0 and len(t['history'])==200
    emit('OWN TOY RESULT',{'name':name,'parameters':t['parameters'],'seconds':t['seconds'],'evaluation':e})
emit('FINAL','All bounded probes and original-record checks completed. GPU records were audited, not replicated.')
