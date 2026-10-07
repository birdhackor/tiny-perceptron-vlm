import torch
a=torch.tensor([[.1,.2,0.]]).repeat(12,1)
b=torch.cat([torch.tensor([[.1,0.,0.]]).repeat(6,1),torch.tensor([[.1,.2,0.]]).repeat(6,1)])
for s in [a,b]:print('counts',torch.bincount(s.softmax(-1).topk(1,-1).indices.flatten(),minlength=3).tolist())
