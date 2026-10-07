import sys,json,torch
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'CPU'}))

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

# Owner supplied proportionate control
print('stored difference after zero',difference.item(),difference.item()>0)
new_difference=(projector(a)-projector(b)).norm()
print('new difference after zero',new_difference.item(),new_difference.item()>0)
print('manual row outputs',float(a[0]@torch.tensor([1.,2.,0.,0.])+.5),float(b[0]@torch.tensor([1.,2.,0.,0.])+.5))
