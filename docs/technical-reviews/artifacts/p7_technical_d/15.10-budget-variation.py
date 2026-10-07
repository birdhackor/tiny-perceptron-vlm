import json,torch
from tiny_perceptron.model import TinyLM,Config
r=json.load(open('docs/course-experiments/results/moe.json'))['results']['variants']
for n in ['top1_aux0','top2_aux0']:
 c=r[n]['model']['config'];m=TinyLM(Config(**c));allp=sum(p.numel() for p in m.parameters());e=sum(p.numel() for b in m.blocks for p in b.ffn.experts.parameters());router=sum(p.numel() for b in m.blocks for p in b.ffn.router.parameters());active=allp-e+e//c['experts']*c['top_k'];print(n,'all',allp,'expert',e,'router',router,'other',allp-e-router,'proxy',active,'raw',r[n]['budget'])
for n in [4,8]:print('toy_k1',n,'total',64*n+4*n,'active',64+4*n,'bytes',4*(64*n+4*n))
