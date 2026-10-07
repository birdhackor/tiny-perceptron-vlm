from tiny_perceptron.natural_concepts import swapped_picture
from torch.nn import functional as F
a,b=swapped_picture();print('spatial changed',int((a!=b).any(0).sum()),'changed channel scalars',int((a!=b).sum()),'cell mean',a[:,8:16,0:8].mean((1,2)).tolist())
moved=a.clone();moved[0,8:16,0:4]=0;moved[0,8:16,8:12]=1
p=F.adaptive_avg_pool2d(a[None],(4,4));q=F.adaptive_avg_pool2d(moved[None],(4,4));print('red moved cell summary equal',torch.equal(p,q),'red old/new',p[0,0,1,:2].tolist(),q[0,0,1,:2].tolist())
