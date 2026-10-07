import json,torch
from torch.nn import functional as F
from tiny_perceptron.model import TinyLM,ModelConfig
r=json.load(open('docs/course-experiments/results/modern.json'))['results']['variants']
x=torch.tensor([[2.,2.,2.],[1.,2.,3.]]);eps=1e-5
print('shift_ln_same',torch.allclose(F.layer_norm(x,(3,),eps=eps),F.layer_norm(x+10,(3,),eps=eps)))
print('shift_rms',((x+10)/((x+10).square().mean(-1,keepdim=True)+eps).sqrt()).tolist())
z=torch.zeros(1,3);print('zero',F.layer_norm(z,(3,),eps=eps).tolist(),(z/(z.square().mean(-1,keepdim=True)+eps).sqrt()).tolist())
for name in ['baseline','rmsnorm']:
 v=r[name];m=TinyLM(ModelConfig(**v['model']['config']));print(name,'parameters',sum(p.numel() for p in m.parameters()),'step_ms',v['training']['warm_step_median_seconds']*1000)
 print('norm_biases',[(k,list(p.shape)) for k,p in m.named_parameters() if 'norm' in k and k.endswith('bias')])
 for split in ['validation','test']:
  q=v['heldout'][split];print(name,split,q['nll_sum']/q['effective_tokens'],q['nll'],q['effective_tokens']);assert q['nll_sum']/q['effective_tokens']==q['nll']
print('paramdiff',r['baseline']['model']['parameters']-r['rmsnorm']['model']['parameters'],5*64)
print('torch',torch.__version__)
