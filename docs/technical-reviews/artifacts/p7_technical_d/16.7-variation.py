import torch
print('torch',torch.__version__)
b=torch.tensor([[.5,1.],[1.5,-1.]])
for a in [torch.tensor([[1.001,2.002],[3.003,4.004]])*10,torch.tensor([[1.,2.],[3.,4.]])]:
 full=a@b
 with torch.autocast('cpu',dtype=torch.bfloat16):low=a@b
 print('input',a.tolist(),'full',full.tolist(),'low',low.float().tolist(),'max_abs',(full-low.float()).abs().max().item())
