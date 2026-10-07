import torch
from torch import nn
embedding = nn.Embedding(3,2)
with torch.no_grad():embedding.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[1.,1.]]))
tied=nn.Linear(2,3,bias=False);tied.weight=embedding.weight
copied=nn.Linear(2,3,bias=False)
with torch.no_grad():copied.weight.copy_(embedding.weight)
h=torch.tensor([[2.,1.]])
print("原分數",tied(h).tolist(),copied(h).tolist());print("同一物件",tied.weight is embedding.weight,copied.weight is embedding.weight)
with torch.no_grad():embedding.weight[0,0]+=1
print("改表後",tied(h).tolist(),copied(h).tolist());print('torch',torch.__version__)
