import json
r=json.load(open('docs/course-experiments/results/moe.json'))['results']['variants']['top2_aux0.01']['validation_routing']
l=r['layers'][1]; n=r['effective_input_tokens']
print('tokens',n,'dispatch_denominator',l['dispatch_denominator'])
assert sum(l['dispatch_counts'])==2*n==l['dispatch_denominator']
print('mean_probability',[round(p,4) for p in l['mean_router_probability']])
print('dispatch_counts',l['dispatch_counts'])
print('recounted_load',[round(c/(2*n),4) for c in l['dispatch_counts']])
assert all(abs(c/(2*n)-f)<1e-12 for c,f in zip(l['dispatch_counts'],l['load_fraction']))
