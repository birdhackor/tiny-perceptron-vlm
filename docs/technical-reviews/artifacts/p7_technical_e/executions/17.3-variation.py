import torch
w=torch.tensor([[.7,.2],[-.7,1.2]]);r=(w/.5).round()*.5;x=torch.tensor([[1.,1.],[1.,-1.]])*10
print('weight_mae',(w-r).abs().mean().item())
print('output_difference',(x@(w-r).T).tolist())
assert torch.allclose(x@(w-r).T,torch.tensor([[4.,0.],[0.,-4.]]),atol=1e-5)
