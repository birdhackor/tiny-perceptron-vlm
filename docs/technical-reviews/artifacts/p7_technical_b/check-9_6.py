import torch
should_refuse=torch.tensor([True,False,True,False])
actually_refuse=torch.ones(4,dtype=torch.bool)
refusal_rate=actually_refuse[should_refuse].float().mean()
completion_rate=(~actually_refuse[~should_refuse]).float().mean()
print('該拒絕的拒絕比例',refusal_rate.item())
print('正常題未拒絕比例',completion_rate.item())
assert refusal_rate.item()==1 and completion_rate.item()==0
actually_refuse=torch.tensor([True,False,True,False])
print('variation correct refusal',(actually_refuse[should_refuse].float().mean().item()),'normal not refusal',(~actually_refuse[~should_refuse]).float().mean().item())
actually_refuse=torch.tensor([True,False,True,True])
print('variation one normal not refused',(~actually_refuse[~should_refuse]).float().mean().item())
assert (~actually_refuse[~should_refuse]).float().mean().item()==.5
print('torch',torch.__version__,'CPU',should_refuse.device)
