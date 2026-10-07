import json,math
v=json.load(open('docs/course-experiments/results/moe.json'))['results']['variants']
for n in ['top1_aux0','top1_aux0.01','top2_aux0','top2_aux0.01']:
 d=v[n];print(n,'steps',d['training']['steps'])
 hs=[]
 for l in d['validation_routing']['layers']:
  fs=[c/l['dispatch_denominator'] for c in l['dispatch_counts']];h=-sum(f*math.log(f) if f else 0 for f in fs)/math.log(4);assert abs(h-l['normalized_load_entropy'])<1e-12;hs.append(round(h,5))
 print('normalizedH',hs,'layer2batchaux',d['validation_routing']['layers'][1]['mean_batch_auxiliary'])
 for split in ['validation','test']:
  h=d['heldout'][split];x=h['nll_sum']/h['effective_tokens'];assert abs(x-h['nll'])<1e-12;print(split,round(x,5),'sum/count',h['nll_sum'],h['effective_tokens'])
