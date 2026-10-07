import json,math,platform,torch
from tiny_perceptron.model import masked_loss,loss_sum
a=torch.tensor([[[0.,2.,0.,0.],[0.,0.,0.,0.]]]);labels=torch.tensor([[1,2]])
b=torch.cat([a,torch.zeros(1,3,4)],dim=1);extended=torch.tensor([[1,2,-100,-100,-100]])
before=masked_loss(a,labels);after=masked_loss(b,extended)
assert torch.allclose(before,after)
s,n=loss_sum(a,labels);s2,n2=loss_sum(b,extended)
expected=(math.log(math.exp(2)+3)-2+math.log(4))/2
assert abs(float(before)-expected)<1e-6 and n==n2==2 and s==s2
c=torch.cat([a,torch.randn(1,9,4)],dim=1);cl=torch.tensor([[1,2]+[-100]*9]);more=masked_loss(c,cl);assert torch.allclose(before,more)
wrong=float(s2)/5
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','before':float(before),'after':float(after),'sum':float(s),'counts':[int(n),int(n2)],'hand_expected':expected,'added_nine_random_logits_loss':float(more),'wrong_divide_by_all_five':wrong},indent=2))
