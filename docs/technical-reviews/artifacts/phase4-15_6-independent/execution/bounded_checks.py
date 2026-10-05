"""Short CPU verification of row restoration and current repository method contracts.

No dataset/model downloads, no optimizer, no training, no score re-evaluation.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-15_6-independent/execution'
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from tiny_perceptron.modern import MoEFFN
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.data import pad_batch

torch.set_num_threads(1)
torch.manual_seed(156)
assert torch.version.cuda is None and not torch.cuda.is_available()
data={'environment':{'python':sys.version,'torch':str(torch.__version__),'torch_git_version':torch.version.git_version,'device':'cpu','cuda_build':str(torch.version.cuda)},'checks':{}}

def routes(rows, values):
    result=torch.zeros(3,2)
    returned=result.index_add_(0,rows,values)
    overwritten=torch.zeros_like(result)
    for row,value in zip(rows,values):overwritten[row]=value
    assert returned is result
    return result,overwritten

values=torch.tensor([[1.,1.],[2.,2.],[3.,3.]])
for name,rows,permutation,expected_sum,expected_over in [
 ('original',[2,0,2],[0,1,2],[[2,2],[0,0],[4,4]],[[2,2],[0,0],[3,3]]),
 ('new_rows',[1,0,1],[0,1,2],[[2,2],[4,4],[0,0]],[[2,2],[3,3],[0,0]]),
 ('swapped_completion',[2,0,2],[2,1,0],[[2,2],[0,0],[4,4]],[[2,2],[0,0],[1,1]])]:
    result,over=routes(torch.tensor(rows),values[permutation])
    assert torch.equal(result,torch.tensor(expected_sum,dtype=torch.float32))
    assert torch.equal(over,torch.tensor(expected_over,dtype=torch.float32))
    data['checks'][name]={'sum':result.tolist(),'overwrite':over.tolist(),'returned_same_object':True}

# All six rows choose the same two experts, deliberately maximally imbalanced.
moe=MoEFFN(width=2,experts=4,top_k=2,hidden=4)
with torch.no_grad():moe.router.weight.copy_(torch.tensor([[3.,0.],[2.,0.],[-2.,0.],[-3.,0.]]))
x=torch.tensor([[[1.,.1],[2.,.2],[3.,.3]],[[4.,.4],[5.,.5],[6.,.6]]])
calls=[0]*4
handles=[]
for i,expert in enumerate(moe.experts):
    def hook(module,args,result,i=i):calls[i]+=args[0].shape[0]
    handles.append(expert.register_forward_hook(hook))
out,aux,chosen=moe(x)
for h in handles:h.remove()
flat=x.reshape(-1,2)
prob=moe.router(flat).softmax(-1);weights=prob.gather(-1,chosen);weights=weights/weights.sum(-1,keepdim=True)
reference=torch.zeros_like(flat)
for row in range(flat.shape[0]):
    for slot in range(2):reference[row]+=moe.experts[int(chosen[row,slot])](flat[row:row+1])[0]*weights[row,slot]
error=float((out.reshape(-1,2)-reference).abs().max())
assert error<=1e-6 and calls==[6,6,0,0] and chosen.tolist()==[[0,1]]*6
data['checks']['dropless_and_shape']={'input_shape':list(x.shape),'output_shape':list(out.shape),'chosen':chosen.tolist(),'expert_input_rows':calls,'selected_jobs':12,'observed_jobs':sum(calls),'reference_max_abs_error':error,'tolerance':1e-6}

# Compile only the already-inspected original methods, not the experiment runner.
path=ROOT/'scripts/course_experiments/architecture.py'
tree=ast.parse(path.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_forward','_routing']]
namespace={'torch':torch,'F':F,'math':math,'pad_batch':pad_batch}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path), 'exec'),namespace)
data['method_extraction']={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'methods':[(n.name,n.lineno,n.end_lineno) for n in nodes]}
examples=[(torch.tensor([4,5,6]),torch.tensor([5,6,7])),(torch.tensor([8]),torch.tensor([9]))]
ids,targets,valid=pad_batch(examples)
model=TinyLM(ModelConfig(vocab_size=16,width=4,layers=1,heads=1,max_length=4,experts=4,top_k=2))
captured={};jobs=[]
def ffn_hook(module,args,result):captured.update(input=args[0].detach(),chosen=result[2].detach())
h=model.blocks[0].ffn.register_forward_hook(ffn_hook)
expert_handles=[expert.register_forward_hook(lambda module,args,result:jobs.append(args[0].shape[0])) for expert in model.blocks[0].ffn.experts]
result=namespace['_forward'](model,ids,valid)
h.remove()
for handle in expert_handles:handle.remove()
ffn=model.blocks[0].ffn
p=ffn.router(captured['input'][valid]).float().softmax(-1)
c=captured['chosen'][valid.reshape(-1)]
load=F.one_hot(c,4).float().mean((0,1));manual_aux=4*(load.detach()*p.mean(0)).sum()
aux_error=float((result['auxiliary']-manual_aux).abs().detach())
route=namespace['_routing'](model,examples,SimpleNamespace(device='cpu'))
assert sum(jobs)==12 and int(valid.sum())==4 and aux_error<=1e-7
assert route['effective_input_tokens']==4 and route['padding_excluded'] is True
assert route['layers'][0]['dispatch_denominator']==8
data['checks']['padding_and_masked_auxiliary']={'batch_shape':list(ids.shape),'valid_mask':valid.tolist(),'padded_rows':6,'effective_input_tokens':4,'top_k':2,'executed_expert_jobs':sum(jobs),'statistical_dispatch_denominator':route['layers'][0]['dispatch_denominator'],'masked_auxiliary':float(result['auxiliary'].detach()),'manual_masked_auxiliary':float(manual_aux.detach()),'absolute_error':aux_error,'tolerance':1e-7,'routing_counts':route['layers'][0]['dispatch_counts']}
data['scope']='Original fence plus finite toy perturbations and one untrained TinyLM forward. No optimizer/update/backward, no quality or speed claim; no dataset or model download; no full experiment runner invoked.'
(OUT/'bounded-results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(data,ensure_ascii=False,indent=2))
