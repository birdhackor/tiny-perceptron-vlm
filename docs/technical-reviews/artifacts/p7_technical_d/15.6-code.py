import torch
rows=torch.tensor([2,0,2]); contributions=torch.tensor([[1.,1.],[2.,2.],[3.,3.]])
result=torch.zeros(3,2); result.index_add_(0,rows,contributions);print('相加結果',result.tolist())
overwritten=torch.zeros(3,2)
for row,value in zip(rows,contributions):overwritten[row]=value
print('覆蓋結果',overwritten.tolist())
