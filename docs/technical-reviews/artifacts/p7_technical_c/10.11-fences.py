import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from torch.nn import functional as F

scores = torch.tensor([[2.0, 0.1], [0.4, 1.8]], requires_grad=True)
labels = torch.arange(2)
image_to_text = F.cross_entropy(scores, labels)
text_to_image = F.cross_entropy(scores.T, labels)
loss = (image_to_text + text_to_image) / 2
loss.backward()
print("圖找文", round(image_to_text.item(), 4))
print("文找圖", round(text_to_image.item(), 4))
print("平均", round(loss.item(), 4))
print("梯度符號", torch.sign(scores.grad).tolist())


# Owner supplied proportional check
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

