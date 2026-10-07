import json
r=json.load(open('docs/course-experiments/results/moe.json'));print('seed',r['seed']);v=r['results']['variants']
for n,d in v.items():
 print(n)
 for split in ['validation','test']:
  h=d['heldout'][split];val=h['nll_sum']/h['effective_tokens'];assert abs(val-h['nll'])<1e-12;print(split,round(val,5),'sum',h['nll_sum'],'targets',h['effective_tokens'])
