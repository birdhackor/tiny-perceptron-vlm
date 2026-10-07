import json,torch
from torch.nn import functional as F
from tiny_perceptron.model import TinyLM,ModelConfig
x=torch.tensor([-4.,-2.,0.,2.,4.]);print('variation_GELU',F.gelu(x).round(decimals=4).tolist());print('variation_relu2',F.relu(x).square().tolist())
try:F.gelu(torch.tensor([-4,-2,0,2,4]))
except Exception as e:print('integer_gelu_error',type(e).__name__,str(e))
r=json.load(open('docs/course-experiments/results/modern.json'))['results']['variants']
for name in ['baseline','relu2']:
 v=r[name];print(name,'parameters',sum(p.numel() for p in TinyLM(ModelConfig(**v['model']['config'])).parameters()),'steps',v['training']['steps'],'median_ms',v['training']['warm_step_median_seconds']*1000,'train_tokens',v['training']['effective_tokens'])
 for split in ['validation','test']:
  z=v['heldout'][split];print(name,split,z['nll_sum']/z['effective_tokens'],z['effective_tokens']);assert z['nll_sum']/z['effective_tokens']==z['nll']
 if name=='relu2':print('test_first',v['heldout']['test']['samples'][0]['prompt'],repr(v['heldout']['test']['samples'][0]['generated']))
print('torch',torch.__version__)
