import torch
x=torch.tensor([[1.,2.]])
w=torch.zeros(2,3)
p=(x@w).softmax(-1)
y=(p[:,:,None]*torch.stack([x,2*x,3*x],dim=1)).sum(1)
print('proportions',p.tolist(),'mixed',y.tolist())
