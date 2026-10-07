import torch,json
from torch import nn
from tiny_perceptron.alignment import LoRALinear
torch.manual_seed(0);layer=LoRALinear(nn.Linear(4,3),rank=2,alpha=2);x=torch.randn(2,4)
with torch.no_grad():
 layer.b.fill_(.1);adapter_a={'a':layer.a.clone(),'b':layer.b.clone()};original=layer(x).clone()
 layer.a.fill_(.2);layer.b.fill_(-.1);adapter_b={'a':layer.a.clone(),'b':layer.b.clone()}
 layer.a.copy_(adapter_a['a']);layer.b.copy_(adapter_a['b'])
print('切回A的B平均',round(layer.b.mean().item(),2));print('A兩個矩陣都恢復',torch.equal(layer.a,adapter_a['a']) and torch.equal(layer.b,adapter_a['b']))
assert torch.equal(layer(x),original)
with torch.no_grad():
 layer.a.copy_(adapter_b['a']);layer.b.copy_(adapter_b['b']);layer.b.copy_(adapter_a['b'])
 wrong=(layer(x)-original).abs().max().item();assert wrong>0
print(json.dumps({'b_average':round(adapter_a['b'].mean().item(),2),'full_restore_output_exact':True,'only_b_restore_max_difference':wrong,'fixed_rank':2,'fixed_alpha':2},indent=2))
