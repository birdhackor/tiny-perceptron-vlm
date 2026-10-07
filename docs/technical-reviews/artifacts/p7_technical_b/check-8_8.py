import torch,json
from torch import nn
from tiny_perceptron.alignment import LoRALinear
torch.manual_seed(0)
layer=LoRALinear(nn.Linear(16,12),rank=2,alpha=2)
x=torch.randn(3,16)
print('可訓練參數',sum(p.numel() for p in layer.parameters() if p.requires_grad))
print('初始相同',torch.equal(layer(x),layer.base(x)))
layer(x).sum().backward()
print('A梯度為零',layer.a.grad.abs().max().item()==0)
print('B梯度非零',layer.b.grad.abs().max().item()>0)
assert sum(p.numel() for p in layer.parameters() if p.requires_grad)==56
assert layer.a.grad.count_nonzero()==0 and layer.b.grad.count_nonzero()>0
assert all(p.grad is None for p in layer.base.parameters())
out={'a_shape':list(layer.a.shape),'b_shape':list(layer.b.shape),'w_shape':list(layer.base.weight.shape),'a_grad_max':layer.a.grad.abs().max().item(),'b_grad_max':layer.b.grad.abs().max().item(),'base_grad_none':True,'environment':{'torch':torch.__version__}}
for rank in [1,4]:
 torch.manual_seed(0);l=LoRALinear(nn.Linear(16,12),rank=rank,alpha=2)
 n=sum(p.numel() for p in l.parameters() if p.requires_grad);assert n==28*rank
 out['rank'+str(rank)]={'parameters':n,'scaling':l.alpha/l.rank,'initial_equal':torch.equal(l(x),l.base(x))}
print(json.dumps(out,indent=2))
