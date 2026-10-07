import torch
for p in [torch.tensor([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]),torch.tensor([[.8,.1,.1]]*3),torch.ones(3,3)/3]:
 f=torch.bincount(p.argmax(-1),minlength=3).float()/3;aux=3*(f*p.mean(0)).sum();print('load',f.tolist(),'aux',aux.item(),'coefficient1_total',(.5+aux).item())
