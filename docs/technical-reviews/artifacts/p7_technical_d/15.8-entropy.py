import math,json
counts=[8536,25630,4875,215];fractions=[c/sum(counts) for c in counts];terms=[-f*math.log(f) if f>0 else 0. for f in fractions];print(round(sum(terms)/math.log(len(counts)),5))
r=json.load(open('docs/course-experiments/results/moe.json'))['results']['variants']['top1_aux0']['validation_routing']['layers'][1]
print('fractions',fractions,'percent',[round(f*100,2) for f in fractions],'entropy',sum(terms),'normalized',sum(terms)/math.log(4))
assert counts==r['dispatch_counts'];assert abs(sum(terms)-r['load_entropy_nats'])<1e-12
for fs in [[.25]*4,[1.,0.,0.,0.]]:print('edge',fs,'H',sum(-f*math.log(f) if f else 0 for f in fs))
