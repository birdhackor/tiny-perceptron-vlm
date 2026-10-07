import torch
q=torch.tensor([.5,.4,.1]);targets=[('原答案',torch.tensor([1.,0.,0.])),('教師硬答案',torch.tensor([1.,0.,0.])),('教師分布',torch.tensor([.7,.2,.1]))]
for name,target in targets:
 logits=q.log().detach().clone().requires_grad_();loss=-(target*logits.log_softmax(0)).sum();loss.backward();print(name,'CE',round(loss.item(),4),'分數梯度',logits.grad.round(decimals=4).tolist())
