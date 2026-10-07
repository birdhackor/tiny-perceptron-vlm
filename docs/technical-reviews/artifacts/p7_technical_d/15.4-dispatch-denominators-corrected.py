import json
v=json.load(open('docs/course-experiments/results/moe.json'))['results']['variants']
for name,d in v.items():
 r=d.get('validation_routing')
 if r is None:print(name,'no router');continue
 n=r['effective_input_tokens'];k=d['model']['config']['top_k'];print(name,'n',n,'k',k,'PADexcluded',r['padding_excluded'])
 for i,l in enumerate(r['layers']):
  assert sum(l['dispatch_counts'])==l['dispatch_denominator']==n*k
  print('layer',i,'counts',l['dispatch_counts'],'sum',sum(l['dispatch_counts']))
