import torch
q=torch.tensor([.6,.2,.2])
for p in [torch.tensor([1.,0.,0.]),torch.tensor([.8,.15,.05]),torch.tensor([.15,.8,.05])]:
 z=q.log().detach().clone().requires_grad_();L=-(p*z.log_softmax(0)).sum();L.backward();print('target',p.tolist(),'grad',z.grad.tolist());assert torch.allclose(z.grad,q-p,atol=1e-7)
