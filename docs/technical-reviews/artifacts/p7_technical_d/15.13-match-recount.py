import json,torch
from tiny_perceptron.model import TinyLM,ModelConfig
v=json.load(open('docs/course-experiments/results/moe.json'))['results']['variants']
for name,d in v.items():
 t=d['training'];print(name,'params',d['budget'],'steps/updates/targets',t['steps'],t['optimizer_updates'],t['effective_tokens'])
 h=d['heldout']['validation'];n=h['nll_sum']/h['effective_tokens'];print('val',round(n,5),'sum/count',h['nll_sum'],h['effective_tokens']);assert abs(n-h['nll'])<1e-12
for target in [142080,208256,340608]:
 candidates=[(abs(sum(p.numel() for p in TinyLM(ModelConfig(width=w,layers=2,heads=4)).parameters())-target),w,sum(p.numel() for p in TinyLM(ModelConfig(width=w,layers=2,heads=4)).parameters())) for w in range(16,161,4)];print('matching',target,min(candidates),'percent',min(candidates)[0]/target*100)
for k in [1,2]:
 total=8*16+4*8;active=k*16+4*8;print('variation8',k,total,active,total/8,active/8)
