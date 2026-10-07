import torch
a=torch.tensor([1.,2.]);b=torch.tensor([1.01,2.]);print('MAE',round((a-b).abs().mean().item(),4),'argmax',a.argmax().item(),b.argmax().item())
