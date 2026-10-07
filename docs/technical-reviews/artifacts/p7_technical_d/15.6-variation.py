import torch
c=torch.tensor([[1.,1.],[2.,2.],[3.,3.]])
r=torch.tensor([1,0,1])
for order in [[0,1,2],[2,1,0]]:
 a=torch.zeros(3,2);a.index_add_(0,r[order],c[order]);b=torch.zeros(3,2)
 for i in order:b[r[i]]=c[i]
 print(order,'sum',a.tolist(),'overwrite',b.tolist())
