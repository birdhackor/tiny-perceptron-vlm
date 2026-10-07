import torch
p=torch.tensor([[.1,.2,.7],[.6,.3,.1]])
for k in (2,3):
 w,i=p.topk(k,-1); print(k,i.tolist(),w.tolist(),w.sum(-1).tolist())
s=torch.tensor([0.,1.,2.],requires_grad=True); w,i=s.softmax(-1).topk(2); w.sum().backward(); print('indices_requires_grad',i.requires_grad,'score_gradient',s.grad.tolist())
