import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from torch.nn import functional as F

image = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
text = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
similarity = F.normalize(image, dim=-1) @ F.normalize(text, dim=-1).T
print(similarity)
print("每張圖選文字欄", similarity.argmax(-1).tolist())


# Owner supplied proportional check
swapped=F.normalize(image,dim=-1)@F.normalize(text.flip(0),dim=-1).T
print('swapped text',swapped.tolist(),swapped.argmax(-1).tolist())
print('normalize 3,4',F.normalize(torch.tensor([3.,4.]),dim=-1).tolist())
ties=F.normalize(torch.tensor([[1.,1.]]),dim=-1)@F.normalize(image,dim=-1).T
print('two same scores sum',ties.tolist(),ties.sum().item())

