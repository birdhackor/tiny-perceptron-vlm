import math, pathlib, hashlib
print('hand directional',sum(math.log1p(math.exp(-d)) for d in [1.9,1.4])/2,sum(math.log1p(math.exp(-d)) for d in [1.6,1.7])/2)
z=scores.detach().clone().requires_grad_(True)
lw=(2*F.cross_entropy(z,labels)+F.cross_entropy(z.T,labels))/3;lw.backward()
print('weighted',lw.item(),tuple(z.grad.shape),z.grad.tolist())
p=pathlib.Path('docs/course-experiments/results/contrastive.json'); raw=p.read_bytes()
pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/contrastive-raw.json').write_bytes(raw)
r=json.loads(raw)
print('rawsha',hashlib.sha256(raw).hexdigest(),'history',r['revision'],r['device'],r['torch_version'])
for n,v in r['results']['variants'].items():
 for s in ['validation','test']:
  x=v[s]
  for key,den,acc in [('image_samples','image_queries','image_to_text_accuracy'),('text_samples','text_queries','text_to_image_accuracy')]:
   count=sum(t['correct'] for t in x[key]);print('recount',n,s,key,count,len(x[key]),x[den],x[acc]);assert count/len(x[key])==x[acc]
