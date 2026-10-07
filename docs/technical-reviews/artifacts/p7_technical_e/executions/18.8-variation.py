import torch
from tiny_perceptron.alignment import distillation_kl
s=torch.zeros(1,3,2,requires_grad=True);p=torch.tensor([.8,.2]);t=p.expand(1,3,2).log().clone().requires_grad_();labels=torch.tensor([[0,0,1]]);L=distillation_kl(s,t,labels,temperature=1);L.backward();print('KL',L.item(),'grad',s.grad.tolist(),'teachergrad',t.grad)
q=s.detach().softmax(-1);H=-(p*p.log()).sum();CE=-(p*q.log()).sum(-1).mean();print('CE',CE.item(),'entropy',H.item(),'CE-H',(CE-H).item());assert torch.allclose(L.detach(),CE-H,atol=1e-7)
ce=-(p*s.log_softmax(-1)).sum(-1).mean();g=torch.autograd.grad(ce,s)[0];print('CE/KLgradient identical',torch.allclose(g,s.grad,atol=1e-7))
