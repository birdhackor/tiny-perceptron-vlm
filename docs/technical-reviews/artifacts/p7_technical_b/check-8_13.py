import torch,json
from torch import nn
from tiny_perceptron.alignment import LoRALinear
torch.manual_seed(0);layer=LoRALinear(nn.Linear(4,3),rank=2,alpha=2)
with torch.no_grad():layer.b.fill_(1.)
delta1=layer.merged_weight()-layer.base.weight;layer.alpha=4;delta2=layer.merged_weight()-layer.base.weight
error=(delta2-2*delta1).abs().max().item();ok=torch.allclose(delta2,2*delta1,atol=1e-6,rtol=0)
print('加倍誤差',error);print('符合兩倍',ok);assert ok
layer.alpha=1;half=layer.merged_weight()-layer.base.weight;assert torch.allclose(half,.5*delta1,atol=1e-6,rtol=0)
print(json.dumps({'alpha1_half_error':(half-.5*delta1).abs().max().item(),'fixed_rank':2,'fixed_AB':True,'scalar_W0.7_BA0.3':{'alpha2':.7+.3,'alpha4':.7+.6,'alpha1':.7+.15}}))
