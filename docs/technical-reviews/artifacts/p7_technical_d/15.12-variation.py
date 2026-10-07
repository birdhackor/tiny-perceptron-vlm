import math,torch
c=torch.tensor([5,1]);n=6;k=1;e=2
for factor in [2/3,1,2]:
 cap=math.ceil(factor*n*k/e);o=(c-cap).clamp(min=0);print('factor',factor,'capacity',cap,'overflow',o.tolist(),'overflow_rate',o.sum().item()/(n*k))
