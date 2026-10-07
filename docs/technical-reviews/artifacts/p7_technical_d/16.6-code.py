import torch
print('torch',torch.__version__)
x=torch.tensor([1.,2.,3.]);whole=torch.tensor(1.,requires_grad=True);(whole*x).square().mean().backward()
accumulated=torch.tensor(1.,requires_grad=True);wrong=torch.tensor(1.,requires_grad=True)
for part in (x[:1],x[1:]):
 ((accumulated*part).square().sum()/len(x)).backward();((wrong*part).square().mean()/2).backward()
print("整批梯度",round(whole.grad.item(),4));print("正確累積",round(accumulated.grad.item(),4));print("錯誤平均",round(wrong.grad.item(),4))
v=torch.tensor(1.,requires_grad=True)
for part in (x[:2],x[2:]):((v*part).square().sum()/len(x)).backward()
print('variation split2/1',v.grad.item());assert torch.allclose(whole.grad,v.grad);assert torch.allclose(whole.grad,accumulated.grad)
