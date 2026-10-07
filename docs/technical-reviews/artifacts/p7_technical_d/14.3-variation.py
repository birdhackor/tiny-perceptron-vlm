import torch
import torch.nn.functional as F
q=torch.tensor([[1.,1.]]);k=torch.tensor([[1.,0.],[0.,.1]])
raw=q@k.T;unit=F.normalize(q,dim=-1)@F.normalize(k,dim=-1).T
print('raw',raw.tolist(),'weights',raw.softmax(-1).tolist());print('unitweights',unit.softmax(-1).tolist());print('unscaled',torch.tensor([1.,0.]).softmax(-1).tolist());print('torch',torch.__version__)
