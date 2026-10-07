import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
print('torch',torch.__version__)
torch.manual_seed(0);layer=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,4));base=torch.randn(2,4)
x_plain=base.clone().requires_grad_();x_recompute=base.clone().requires_grad_();y_plain=layer(x_plain);y_recompute=checkpoint(layer,x_recompute,use_reentrant=False)
y_plain.square().sum().backward();y_recompute.square().sum().backward()
print("輸出形狀",tuple(y_recompute.shape));print("輸出最大差",(y_plain-y_recompute).abs().max().item());print("輸入梯度最大差",(x_plain.grad-x_recompute.grad).abs().max().item());assert torch.allclose(x_plain.grad,x_recompute.grad,atol=1e-6)
