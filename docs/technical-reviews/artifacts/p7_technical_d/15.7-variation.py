import math,torch
for norm in [False,True]:
 s=torch.tensor([math.log(2),0.],requires_grad=True);p=s.softmax(-1).topk(1).values[0];y=(p/p if norm else p);y.square().backward();print(norm,'output',y.item(),'gradient',s.grad.tolist())
print('norm3_4',torch.tensor([3.,4.]).norm().item())
