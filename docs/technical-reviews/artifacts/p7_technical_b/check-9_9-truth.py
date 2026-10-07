import torch
scores=torch.tensor([[2.,1.,0.]])
for scale in [1.,10.]:
    p=(scores*scale).softmax(-1);c,i=p.max(-1)
    print(scale,round(c.item(),4),i.item(),i.item()==0)
    assert i.item()==0
