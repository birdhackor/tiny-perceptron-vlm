import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from torch.nn import functional as F

scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
targets = torch.arange(2)
loss = F.cross_entropy(scores / 0.5, targets)
loss.backward()
print("代價", round(loss.item(), 4))
print("梯度", scores.grad)


# Owner supplied proportional check
import math
print('hand loss', math.log1p(math.exp(-4)))
print('hand gradient',1/(1+math.exp(4)))
print('hand step',2+0.1/(1+math.exp(4)),-0.1/(1+math.exp(4)))
wrong_scores=scores.detach().clone().requires_grad_(True)
wrong_loss=F.cross_entropy(wrong_scores/.5,torch.tensor([1,0])); wrong_loss.backward()
print('reversed',wrong_loss.item(),wrong_scores.grad.tolist())

