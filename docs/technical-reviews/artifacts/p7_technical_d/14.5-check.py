import json,torch
from torch.nn import functional as F
from tiny_perceptron.model import TinyLM,ModelConfig
content=torch.tensor([2.,2.,2.]);gate=torch.zeros(3);print('zero', (content*F.silu(gate)).tolist());gate[0]=2;print('firstonly',(content*F.silu(gate)).tolist())
r=json.load(open('docs/course-experiments/results/modern.json'))['results']['variants']
for name in ['baseline','swiglu']:
 v=r[name];m=TinyLM(ModelConfig(**v['model']['config']));print(name,'parameters',sum(p.numel() for p in m.parameters()),'ffnhidden',m.blocks[0].ffn.up.out_features,'steps',v['training']['steps'],'median_ms',v['training']['warm_step_median_seconds']*1000)
 for split in ['validation','test']:
  z=v['heldout'][split];print(name,split,z['nll_sum']/z['effective_tokens'],z['effective_tokens']);assert z['nll_sum']/z['effective_tokens']==z['nll']
print('extra',174848-141568,2*(64*256+256));print('torch',torch.__version__)
