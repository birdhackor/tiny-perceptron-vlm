import torch
from tiny_perceptron.alignment import distillation_loss
for a in [0.,.25,.5,1.]:
 s=torch.zeros(1,2,2,requires_grad=True);t=torch.tensor([[[.8,.2],[.8,.2]]]).log();labels=torch.tensor([[0,1]]);L=distillation_loss(s,t,labels,alpha=a,temperature=1);L.backward();print('alpha',a,'loss',L.item(),'grad',s.grad.tolist())
