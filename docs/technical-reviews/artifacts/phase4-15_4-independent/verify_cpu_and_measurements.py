"""Independent bounded CPU checks for section 15.4; no training or downloads."""
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/technical-reviews/artifacts/phase4-15_4-independent'
sys.path.insert(0,str(ROOT))
import torch
from tiny_perceptron.modern import MoEFFN
from tiny_perceptron.data import pad_batch
from scripts.course_experiments.common import split_records, records_sha256, text_examples

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
environment={
    'python':sys.version,'python_executable':sys.executable,
    'torch':str(torch.__version__),'torch_git':str(torch.version.git_version),
    'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),
    'cpu_threads':str(torch.get_num_threads()),'cwd':str(Path.cwd()),
    'offline_flags':{k:os.environ.get(k,'unset') for k in ('CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE')},
}
(OUT/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')

# Execute precisely the extracted source fence without changing its bytes.
original=OUT/'code/original-fence-1.py'
fence_run=subprocess.run([sys.executable,str(original)],cwd=ROOT,check=True,capture_output=True)
(OUT/'original-fence.stdout.txt').write_bytes(fence_run.stdout)
(OUT/'original-fence.stderr.txt').write_bytes(fence_run.stderr)
assert hashlib.sha256(original.read_bytes()).hexdigest()=='b447939f95a05b48aef7c64bdcc9fb46168e02a7172f3d2424d285fd0d0f96c8'
print('ORIGINAL FENCE STDOUT')
print(fence_run.stdout.decode(),end='')

checks={}
prob=torch.tensor([[.1,.7,.2],[.6,.3,.1]])
original_cases=[]
for k,expected_chosen,expected_sums in ((2,[[1,2],[0,1]],[.9,.9]),(1,[[1],[0]],[.7,.6])):
    weights,chosen=prob.topk(k,dim=-1)
    assert chosen.tolist()==expected_chosen
    assert torch.allclose(weights.sum(-1),torch.tensor(expected_sums),atol=1e-7,rtol=0)
    assert weights.shape==chosen.shape==(2,k)
    original_cases.append({'k':k,'chosen':chosen.tolist(),'weights':weights.tolist(),'sums':weights.sum(-1).tolist(),'index_dtype':str(chosen.dtype)})
changed=prob.clone();changed[0]=torch.tensor([.1,.2,.7])
weights,chosen=changed.topk(2,dim=-1)
assert chosen[0].tolist()==[2,1]
assert torch.allclose(weights[0],torch.tensor([.7,.2]),atol=1e-7,rtol=0)
all_weights,all_chosen=changed.topk(3,dim=-1)
assert torch.allclose(all_weights.sum(-1),torch.ones(2),atol=1e-7,rtol=0)
checks['topk_axes_values_indices']={'original':original_cases,'swapped_first_row':{'chosen':chosen.tolist(),'weights':weights.tolist()},'k3':{'chosen':all_chosen.tolist(),'sums':all_weights.sum(-1).tolist()},'tolerance':'absolute 1e-7 float32; exact integer indices for distinct values'}

gradient_cases=[]
for k in (1,2):
    scores=torch.tensor([math.log(.1),math.log(.7),math.log(.2)],dtype=torch.float64,requires_grad=True)
    selected,indices=scores.softmax(-1).topk(k,dim=-1)
    coefficients=torch.arange(1,k+1,dtype=torch.float64)
    objective=(selected*coefficients).sum()
    actual=torch.autograd.grad(objective,scores)[0]
    eps=1e-6
    approximations=[]
    for j in range(3):
        high=scores.detach().clone();low=scores.detach().clone()
        high[j]+=eps;low[j]-=eps
        hw,hi=high.softmax(-1).topk(k,dim=-1)
        lw,li=low.softmax(-1).topk(k,dim=-1)
        assert torch.equal(hi,indices) and torch.equal(li,indices)
        approximations.append(float(((hw*coefficients).sum()-(lw*coefficients).sum())/(2*eps)))
    approximation=torch.tensor(approximations,dtype=torch.float64)
    error=float((actual-approximation).abs().max())
    assert error<1e-8 and float(actual.abs().sum())>0
    assert indices.dtype==torch.int64 and not indices.requires_grad
    gradient_cases.append({'k':k,'chosen':indices.tolist(),'values_requires_grad':selected.requires_grad,'indices_requires_grad':indices.requires_grad,'autograd':actual.tolist(),'finite_difference':approximations,'max_abs_error':error})
left=torch.tensor([.4999,.5001,.0]).topk(1).indices.item()
right=torch.tensor([.5001,.4999,.0]).topk(1).indices.item()
assert (left,right)==(1,0)
checks['gradient_local_continuity']={'cases':gradient_cases,'crossing_rank_boundary':{'left_chosen':left,'right_chosen':right},'support':'Gradients through retained softmax values away from rank ties; this does not differentiate integer identity or perform any parameter update.'}

dispatch_cases=[]
for k in (1,2):
    torch.manual_seed(42)
    module=MoEFFN(width=2,experts=4,top_k=k,hidden=3).double().eval()
    with torch.no_grad():
        module.router.weight.copy_(torch.tensor([[0.,4.],[4.,3.],[3.,0.],[-4.,-4.]],dtype=torch.float64))
    rows=[0,0,0,0];calls=[0,0,0,0]
    handles=[]
    def counter(i):
        def record(_module,args):
            rows[i]+=len(args[0]);calls[i]+=1
        return record
    for i,expert in enumerate(module.experts):handles.append(expert.register_forward_pre_hook(counter(i)))
    try:
        with torch.no_grad():output,_,chosen=module(torch.eye(2,dtype=torch.float64))
    finally:
        for h in handles:h.remove()
    expect=[[1],[0]] if k==1 else [[1,2],[0,1]]
    expect_rows=[1,1,0,0] if k==1 else [1,2,1,0]
    assert chosen.tolist()==expect and rows==expect_rows and sum(rows)==2*k
    assert output.shape==(2,2)
    dispatch_cases.append({'k':k,'chosen':chosen.tolist(),'expert_calls':calls,'expert_input_rows':rows,'total_expert_input_rows':sum(rows),'dense_all_expert_input_rows':8,'output_shape':list(output.shape)})
checks['real_sparse_execution']={'cases':dispatch_cases,'support':'Short CPU forward-hook observation on current and original-hash-matching MoEFFN; verifies skipping unselected experts, without a runtime speed claim.'}

raw_result=OUT/'inputs/moe-original-result-opaque.json'
data=json.loads(raw_result.read_text())
pointers=[]
def get(pointer):
    value=data
    for key in pointer.strip('/').split('/'):
        value=value[int(key)] if isinstance(value,list) else value[key]
    pointers.append(pointer)
    return value
provenance={key:get('/'+key) for key in ('schema_version','experiment_id','revision','device','seed','torch_version','python_version','step_scale')}
provenance['original_implementation_hashes']={key:get('/code_sha256/'+key.replace('~','~0').replace('/','~1')) for key in []}
# JSON object keys with slashes are accessed directly below, with RFC6901 pointers logged.
for key in ('tiny_perceptron/modern.py','tiny_perceptron/data.py','tiny_perceptron/model.py','scripts/course_experiments/common.py','scripts/course_experiments/architecture.py'):
    pointers.append('/code_sha256/'+key.replace('~','~0').replace('/','~1'))
    provenance['original_implementation_hashes'][key]=data['code_sha256'][key]
dataset_validation={'records':get('/results/dataset/validation/records'),'sha256':get('/results/dataset/validation/sha256')}
routing=[]
for variant in ('top1_aux0','top1_aux0.01','top2_aux0','top2_aux0.01'):
    base='/results/variants/'+variant
    cfg={key:get(base+'/model/config/'+key) for key in ('layers','experts','top_k')}
    n=get(base+'/validation_routing/effective_input_tokens')
    excluded=get(base+'/validation_routing/padding_excluded')
    assert cfg['layers']==2 and cfg['experts']==4 and n==39256 and excluded is True
    layer_records=[]
    for i in range(cfg['layers']):
        lp=base+'/validation_routing/layers/'+str(i)
        counts=get(lp+'/dispatch_counts');denominator=get(lp+'/dispatch_denominator');fractions=get(lp+'/load_fraction')
        assert len(counts)==len(fractions)==4 and all(type(c) is int for c in counts)
        assert denominator==sum(counts)==n*cfg['top_k']
        recomputed=[c/denominator for c in counts]
        error=max(abs(a-b) for a,b in zip(fractions,recomputed))
        assert error<1e-15 and abs(sum(fractions)-1)<1e-15
        layer_records.append({'layer':i,'dispatch_counts':counts,'dispatch_denominator':denominator,'load_fraction':fractions,'recomputed_load_fraction':recomputed,'max_abs_error':error})
    routing.append({'variant':variant,'config':cfg,'effective_input_tokens':n,'padding_excluded':excluded,'layers':layer_records})

# Recreate only the CPU data split and padding counts from existing raw stories.
stories=[json.loads(line) for line in (OUT/'inputs/tinystories-original-512.jsonl').read_text().splitlines()]
for record in stories:record['family']=record.get('text_sha256',records_sha256([{'text':record['text']}]))
split=split_records(stories,provenance['seed'])
assert len(split['validation'])==dataset_validation['records']==51
assert records_sha256(split['validation'])==dataset_validation['sha256']
examples=text_examples(split['validation'],max_length=128)
valid_count,padded_count=0,0
for start in range(0,len(examples),16):
    x,_,valid=pad_batch(examples[start:start+16])
    valid_count+=int(valid.sum());padded_count+=x.numel()
assert valid_count==sum(len(x) for x,_ in examples)==39256
checks['original_routing_measurements']={
    'original_result_sha256':hashlib.sha256(raw_result.read_bytes()).hexdigest(),
    'provenance':provenance,'inspected_json_pointers':pointers,
    'validation_dataset':dataset_validation,'measurements':routing,
    'independent_input_denominator_reconstruction':{'validation_records':len(split['validation']),'validation_records_sha256':records_sha256(split['validation']),'chunk_examples':len(examples),'effective_input_positions':valid_count,'padded_input_positions':padded_count,'excluded_padding_positions':padded_count-valid_count,'batch_size':16,'max_chunk_length':128},
    'support':'Recalculation of preserved result counts, shares, and original validation-input denominator; no training, model loading, GPU evaluation, or quality comparison.'
}
(OUT/'verification-results.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))
