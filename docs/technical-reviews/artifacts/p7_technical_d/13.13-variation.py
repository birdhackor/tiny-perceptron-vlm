import torch
from tiny_perceptron.posttraining import ppo_clipped_objective
r=torch.tensor([.7,1.,1.3,.7,1.,1.3],requires_grad=True)
a=torch.tensor([1.,1.,1.,-1.,-1.,-1.]); old=torch.full((6,),.5).log()
x=ppo_clipped_objective(old+r.log(),old,a,.1);(-x['surrogate'].sum()).backward()
print('surrogate',x['surrogate'].tolist());print('gradient',r.grad.tolist());print('torch',torch.__version__)
