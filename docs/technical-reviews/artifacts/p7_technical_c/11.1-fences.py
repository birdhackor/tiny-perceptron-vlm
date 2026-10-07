import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from torch import nn

torch.manual_seed(0)
projector = nn.Linear(4, 8)
a, b = torch.zeros(1, 4), torch.ones(1, 4)
difference = (projector(a) - projector(b)).norm()
print("輸出形狀", tuple(projector(a).shape))
print("對輸入差異敏感", difference.item() > 0)


print('fence 1')
with torch.no_grad():
    projector.weight.zero_()


# Owner supplied proportional check
print('zero new difference',(projector(a)-projector(b)).norm().item(),'old difference',difference.item())
w=torch.tensor([1.,2.,0.,0.]);print('manual affine',float(w@a[0]+.5),float(w@b[0]+.5))

