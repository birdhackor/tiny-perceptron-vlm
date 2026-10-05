"""Bounded CPU checks for section 15.5: arithmetic, routing convention, existing update counters."""
from pathlib import Path
import sys, json, hashlib
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from tiny_perceptron.modern import MoEFFN
A=Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device('cpu')
assert torch.version.cuda is None and not torch.cuda.is_available()
w=torch.tensor([.7,.2],dtype=torch.float64)
o=torch.tensor([[1.,0.],[0.,2.]],dtype=torch.float64)
n=w/w.sum()
r={'environment':{'python':sys.version,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'device':'cpu','cuda_build':str(torch.version.cuda)},'arithmetic':{}}
for label,value,expected in [('direct',w@o,[.7,.4]),('normalized_weights',n,[7/9,2/9]),('normalized_output',n@o,[7/9,4/9]),('swap_outputs_only',n@o.flip(0),[2/9,14/9]),('swap_both',n.flip(0)@o.flip(0),[7/9,4/9])]:
 torch.testing.assert_close(value,torch.tensor(expected,dtype=torch.float64),rtol=0,atol=1e-14)
 r['arithmetic'][label]=value.tolist()
assert (n@o).shape==(2,) and torch.cat([o[0],o[1]]).shape==(4,)
assert bool(((n@o)>=o.min(0).values).all()) and bool(((n@o)<=o.max(0).values).all())
r['arithmetic']['output_shape']=list((n@o).shape)
r['arithmetic']['concatenation_shape']=list(torch.cat([o[0],o[1]]).shape)
r['arithmetic']['selected_weight_sum']=float(w.sum())
p=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
coefficient=p/p
coefficient.backward()
assert coefficient.item()==1 and p.grad.item()==0
r['single_gate_normalization']={'coefficient':coefficient.item(),'gradient':p.grad.item()}
class Fixed(nn.Module):
 def __init__(self,v):
  super().__init__();self.register_buffer('v',torch.tensor(v));self.calls=0
 def forward(self,x):self.calls+=len(x);return self.v.expand(len(x),-1)
r['actual_MoEFFN']={}
for k,expected in [(1,[.7,0.]),(2,[7/9,4/9])]:
 m=MoEFFN(2,experts=3,top_k=k,hidden=3)
 with torch.no_grad():
  m.router.weight.zero_();m.router.weight[:,0].copy_(torch.tensor([.1,.7,.2]).log())
 m.experts=nn.ModuleList([Fixed([99.,99.]),Fixed([1.,0.]),Fixed([0.,2.])])
 y,aux,chosen=m(torch.tensor([[1.,0.]]))
 torch.testing.assert_close(y[0],torch.tensor(expected),rtol=0,atol=1e-7)
 assert y.shape==(1,2) and m.experts[0].calls==0
 r['actual_MoEFFN'][str(k)]={'chosen':chosen.tolist(),'output':y.tolist(),'expert_token_calls':[e.calls for e in m.experts]}
# Read only specified original measurement / provenance pointers. Do not use author conclusions.
x=json.loads((A/'inputs/moe-original.json').read_text())
orig=A/'inputs/architecture-original-revision.py'
assert hashlib.sha256(orig.read_bytes()).hexdigest()==x['code_sha256']['scripts/course_experiments/architecture.py']
for path in ['tiny_perceptron/modern.py','tiny_perceptron/model.py']:
 assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==x['code_sha256'][path]
r['provenance']={k:x[k] for k in ['revision','device','seed','torch_version','python_version']}
fields=['requested_steps','steps','optimizer_updates','skipped_updates','effective_tokens','batch_size','auxiliary_weight','scaler_enabled','all_parameters_finite','history']
r['existing_training_measurements']={}
r['inspected_pointers']=['/'+k for k in ['revision','device','seed','torch_version','python_version']]+['/code_sha256/'+p.replace('~','~0').replace('/','~1') for p in ['scripts/course_experiments/architecture.py','tiny_perceptron/modern.py','tiny_perceptron/model.py']]
for name,variant in x['results']['variants'].items():
 t=variant['training'];measure={k:t[k] for k in fields}
 assert measure['requested_steps']==measure['steps']==measure['optimizer_updates']==180
 assert measure['skipped_updates']==0 and measure['effective_tokens']>0
 assert measure['all_parameters_finite'] and all(h['gradients_finite'] for h in measure['history'])
 r['existing_training_measurements'][name]=measure
 r['inspected_pointers'] += ['/results/variants/'+name+'/training/'+k for k in fields]
(A/'probe-result.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'arithmetic':r['arithmetic'],'single_gate_normalization':r['single_gate_normalization'],'actual_MoEFFN':r['actual_MoEFFN'],'update_counts':{k:{f:v[f] for f in ['requested_steps','steps','optimizer_updates','skipped_updates','effective_tokens']} for k,v in r['existing_training_measurements'].items()},'assertions':'passed'},ensure_ascii=False,indent=2))
