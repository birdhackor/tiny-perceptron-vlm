import json,torch,hashlib,math
from tiny_perceptron.model import TinyLM,ModelConfig
r=json.load(open('docs/course-experiments/results/modern.json'))
for p in ['scripts/course_experiments/architecture.py','tiny_perceptron/modern.py','tiny_perceptron/model.py','tiny_perceptron/attention.py','tiny_perceptron/data.py']:
 h=hashlib.sha256(open(p,'rb').read()).hexdigest();print('codehash',p,h,h==r['code_sha256'][p]);assert h==r['code_sha256'][p]
for name in ['baseline','rope']:
 v=r['results']['variants'][name];m=TinyLM(ModelConfig(**v['model']['config']));n=sum(x.numel() for x in m.parameters());print(name,'parameters',n,'steps',v['training']['steps'],'max',m.config.max_length);assert n==v['model']['parameters']
 for split in ['validation','test']:
  x=v['heldout'][split];calc=x['nll_sum']/x['effective_tokens'];print(name,split,calc,x['nll'],x['effective_tokens'],abs(calc-x['nll']));assert abs(calc-x['nll'])<1e-12
 if name=='rope':
  rows=[x for x in v['heldout']['test']['samples'] if x['prompt']=='Once upon a time, there '];print('sample',[(x['prompt'],x['generated']) for x in rows])
print('parameter_difference',141568-133376,128*64)
print('variation22_26',math.cos((26-22)*math.pi/4));print('utf8sizes',len('a'.encode()),len('猫'.encode()));print('torch',torch.__version__)
