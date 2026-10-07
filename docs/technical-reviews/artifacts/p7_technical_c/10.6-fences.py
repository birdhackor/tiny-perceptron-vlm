import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from torch import nn

torch.manual_seed(0)
vision = torch.randn(1, 16, 8)
projector = nn.Linear(8, 12)
language_features = projector(vision)
print("視覺", tuple(vision.shape))
print("文字介面", tuple(language_features.shape))
print("接頭矩陣", tuple(projector.weight.shape))


# Owner supplied proportional check
changed = nn.Linear(8,10)
print('out10',tuple(changed(vision).shape),tuple(changed.weight.shape))
toy=torch.tensor([1.,2.]);transform=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
print('recombine', (transform@toy).tolist())

