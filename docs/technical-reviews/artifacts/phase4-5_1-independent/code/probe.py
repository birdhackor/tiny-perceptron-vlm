"""Bounded CPU boundary and original-evidence audit; no experiment retraining."""
from pathlib import Path
import ast
import hashlib
import json
import math
import os
import platform
import random
import sys

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT/'docs/technical-reviews/artifacts/phase4-5_1-independent'
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.model import TinyLM, ModelConfig, loss_sum, masked_loss
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device('cpu')
def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def state(model): return {name:p.detach().clone() for name,p in model.named_parameters()}
def diff(model, old): return max(float((p.detach()-old[name]).abs().max()) for name,p in model.named_parameters())
environment = dict(python=platform.python_version(), torch=str(torch.__version__), torch_git_version=str(torch.version.git_version), device='cpu', cuda_available=str(torch.cuda.is_available()), cuda_build=str(torch.version.cuda), threads=str(torch.get_num_threads()), cwd=str(Path.cwd()), offline={k:os.environ.get(k,'') for k in ['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS']})
dump('probe-environment.json', environment)
torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=5,width=8,max_length=8))
x=torch.tensor([[1,2,3]]); y=torch.tensor([[2,3,4]])
optimizer=torch.optim.AdamW(model.parameters(),lr=0.03,weight_decay=0.0)
start=state(model)
optimizer_ids={id(p) for group in optimizer.param_groups for p in group['params']}
assert optimizer_ids == {id(p) for p in model.parameters()}
assert all(p.requires_grad for p in model.parameters())
optimizer.zero_grad()
assert all(p.grad is None for p in model.parameters())
logits=model(x)['logits']
total,count=loss_sum(logits,y)
loss=masked_loss(logits,y)
assert logits.shape==(1,3,5) and model.embedding.weight.shape==(5,8)
assert int(count)==3 and torch.allclose(loss,total/count)
try:
    model(torch.ones((1,9), dtype=torch.long))
except ValueError as error:
    assert '序列超過 max_length' in str(error)
    length_error=str(error)
else:
    raise AssertionError('Configured maximum length 8 was not enforced')
forward_delta=diff(model,start)
loss.backward()
backward_delta=diff(model,start)
grad=model.embedding.weight.grad
manual_norm=grad.double().square().sum().sqrt()
assert torch.isfinite(grad).all() and float(grad.norm())>0
assert torch.allclose(grad.norm().double(),manual_norm,rtol=1e-6,atol=1e-8)
assert grad[0].count_nonzero()==0 and grad[4].count_nonzero()==0
assert forward_delta==0 and backward_delta==0 and len(optimizer.state)==0
before_step=float(loss.detach())
optimizer.step()
step_delta=diff(model,start)
updated_loss=float(masked_loss(model(x)['logits'],y).detach())
state_steps={float(s['step']) for s in optimizer.state.values()}
assert step_delta>0 and state_steps=={1.0}
updated=state(model)
optimizer.zero_grad()
assert all(p.grad is None for p in model.parameters())
optimizer.step()
cleared_step_delta=diff(model,updated)
assert cleared_step_delta==0 and {float(s['step']) for s in optimizer.state.values()}=={1.0}
boundary=dict(logits_shape=list(logits.shape),embedding_shape=list(grad.shape),effective_target_count=int(count),max_length=8,length_9_error=length_error,optimized_parameter_tensors=len(optimizer_ids),optimized_parameter_elements=sum(p.numel() for p in model.parameters()),all_parameters_require_grad=True,zero_grad_default_sets_none=True,forward_parameter_delta=forward_delta,backward_parameter_delta=backward_delta,embedding_gradient_norm=float(grad.norm()),manual_all_40_entries_norm=float(manual_norm),norm_tolerance='rtol=1e-6, atol=1e-8',gradient_min=float(grad.min()),gradient_max=float(grad.max()),unused_embedding_rows_zero=[0,4],step_parameter_delta=step_delta,loss_before_step=before_step,loss_recomputed_after_step=updated_loss,optimizer_state_steps_after_one_step=sorted(state_steps),step_after_clear_parameter_delta=cleared_step_delta,scope='One diagnostic update only. Forward and backward do not update parameters; AdamW.step does. A positive norm is a signal measure, not a positive gradient at every entry or a generalization result.')
dump('boundary-result.json',boundary)

# AST-extract only the original version's deterministic record/split/tokenization helpers.
# No imports or calls to fit_lm, run_text_foundation, or any GPU/train/data-fetch recipe.
original=OUT/'original/experiment-version'
namespace=dict(torch=torch,json=json,random=random,hashlib=hashlib)
selected = {
 'tiny_perceptron/data.py': {'IGNORE','SPECIALS','ByteTokenizer','shifted','pad_batch'},
 'scripts/prepare_data.py': {'generate_records'},
 'scripts/course_experiments/common.py': {'split_records','records_sha256','text_examples'},
}
extracted=[]
for path,names in selected.items():
    raw=(original/path).read_bytes()
    tree=ast.parse(raw,filename=path)
    nodes=[]
    for node in tree.body:
        name=getattr(node,'name',None)
        if isinstance(node,ast.Assign): name=getattr(node.targets[0],'id',None)
        if name in names:
            nodes.append(node)
            extracted.append(dict(path=path,name=name,first_line=node.lineno,last_line=node.end_lineno,input_sha256=sha(raw)))
    assert len(nodes)==len(names),path
    exec(compile(ast.Module(body=nodes,type_ignores=[]),f'original-experiment-version:{path}','exec'),namespace)
raw_report=(OUT/'original/current/docs/course-experiments/results/text_foundation.json').read_bytes()
report=json.loads(raw_report)
parts=namespace['split_records'](namespace['generate_records']('toy-text'),seed=report['seed'])
text_sets={split:{r['text'] for r in records} for split,records in parts.items()}
assert not text_sets['train']&text_sets['validation']
assert not text_sets['train']&text_sets['test']
assert not text_sets['validation']&text_sets['test']
derived=OUT/'derived-data';derived.mkdir(exist_ok=True)
split_checks={};examples_by_split={}
for split,records in parts.items():
    raw=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records).encode('utf-8')
    (derived/f'{split}.jsonl').write_bytes(raw)
    metadata=report['results']['data'][split]
    assert sha(raw)==metadata['sha256'] and len(records)==metadata['records']
    examples=namespace['text_examples'](records,mode='text',max_length=128)
    examples_by_split[split]=examples
    n=sum(int((labels!=-100).sum()) for inputs,labels in examples)
    raw_bytes=sum(len(record['text'].encode('utf-8')) for record in records)
    assert n==raw_bytes+len(records)  # all byte targets plus EOS; BOS is input-only.
    split_checks[split]=dict(records=len(records),families=len({r['family'] for r in records}),jsonl_sha256=sha(raw),reported_sha256=metadata['sha256'],matches_recorded_raw_split_hash=True,examples=len(examples),raw_utf8_bytes=raw_bytes,effective_targets_per_complete_evaluation=n,includes_eos_targets=len(records))
train=report['results']['training']
assert namespace['records_sha256'](parts['train'])==train['records_sha256']
sampler=random.Random(report['seed'])
exposures=sum(sum(int((y!=-100).sum()) for x,y in sampler.choices(examples_by_split['train'],k=16)) for step in range(train['steps']))
assert exposures==train['effective_tokens']
validation=report['results']['after']['validation']
assert math.isclose(validation['nll'],validation['nll_sum']/validation['effective_tokens'],rel_tol=0,abs_tol=1e-12)
assert validation['effective_tokens']==split_checks['validation']['effective_targets_per_complete_evaluation']
assert validation['records']==1 and validation['examples']==1
for number,display in [(train['initial_loss'],'5.79243'),(train['final_loss'],'0.06352'),(validation['nll'],'0.96282')]:
    assert f'{number:.5f}'==display
assert train['steps']==600 and train['records']==9 and report['gpu']=='NVIDIA L4'
assert report['device']=='cuda' and report['evidence_status']=='complete_run' and report['step_scale']==1.0
audit=dict(original_report_sha256=sha(raw_report),original_revision=report['revision'],original_torch_version=report['torch_version'],original_python_version=report['python_version'],original_device=report['device'],original_gpu=report['gpu'],seed=report['seed'],recorded_steps=train['steps'],batch_size_from_original_fit_lm_default=16,update_recipe_from_original_contract='new_lm width=64,layers=2,max_length=128; fit_lm AdamW lr=.003 default weight decay; forward/backward/clip/step for 600 updates; initial/final _nll evaluate same full training positions',training_evaluation_targets_per_complete_pass=split_checks['train']['effective_targets_per_complete_evaluation'],training_target_exposures_reconstructed_without_model=exposures,reported_training_target_exposures=train['effective_tokens'],initial_train_nll=train['initial_loss'],final_train_nll=train['final_loss'],validation_nll=validation['nll'],validation_nll_sum=validation['nll_sum'],validation_effective_targets=validation['effective_tokens'],rounded_text_values=['5.79243','0.06352','0.96282'],rounding_tolerance='round to 5 decimals, max absolute rounding difference 5e-6',split_checks=split_checks,original_function_extractions=extracted,scope='Audit existing original JSON and matching revision helpers. Raw split bytes reconstructed deterministically and SHA matched to original JSON; no weights loaded, no forward evaluation of the old model, and no training performed.')
dump('empirical-audit.json',audit)
print(json.dumps(dict(boundary=boundary,empirical=audit),ensure_ascii=False,indent=2))
