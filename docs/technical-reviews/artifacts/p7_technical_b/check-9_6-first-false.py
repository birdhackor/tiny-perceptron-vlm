import torch
s=torch.tensor([True,False,True,False]);a=torch.tensor([False,False,True,False])
print('應拒絕',(a[s].float().mean()).item(),'正常未拒',(~a[~s]).float().mean().item())
assert a[s].float().mean().item()==.5 and (~a[~s]).float().mean().item()==1
