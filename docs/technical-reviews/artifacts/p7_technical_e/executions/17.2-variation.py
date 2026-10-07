import torch
x=torch.tensor([-0.7,0.2,0.7,1.2]);s=.25
q=(x/s).round().clamp(-127,127).to(torch.int8);r=q.float()*s
assert q.tolist()==[-3,1,3,5]
print('scale .25 codes',q.tolist(),'restored',r.tolist(),'max_error',(x-r).abs().max().item(),'positive_range',127*s)
y=torch.tensor([-2.5,-1.5,-.5,.5,1.5,2.5]);print('ties',torch.round(y).tolist())
assert torch.round(y).tolist()==[-2,-2,0,0,2,2]
