import json,math,platform,torch
from torch.nn import functional as F
z=torch.tensor([[0.,0.,3.,1.]])
values={str(label):float(F.cross_entropy(z,torch.tensor([label]))) for label in [2,3]}
w=torch.tensor([[0.,0.,3.,5.]]);wrongloss=float(F.cross_entropy(w,torch.tensor([3])));assert wrongloss<values['3']
hand={str(label):math.log(2+math.exp(3)+math.exp(1))-float(z[0,label]) for label in [2,3]};wh=math.log(2+math.exp(3)+math.exp(5))-5
assert all(abs(values[k]-v)<1e-6 for k,v in hand.items()) and abs(wrongloss-wh)<1e-6
gradz=z.clone().requires_grad_();F.cross_entropy(gradz,torch.tensor([3])).backward();assert gradz.grad[0,3]<0
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','label_losses':values,'wrong_favored_loss':wrongloss,'hand_losses':hand,'hand_wrong_favored':wh,'probabilities_before':z.softmax(-1).tolist(),'probabilities_after':w.softmax(-1).tolist(),'wrong_label_gradient':gradz.grad.tolist()},indent=2))
