import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from torch import nn
from tiny_perceptron.multimodal import scene, patchify

torch.manual_seed(0)
patches = patchify(scene()[None], 4)
embedding = nn.Linear(48, 8)
features = embedding(patches)
print("輸入", tuple(patches.shape))
print("輸出", tuple(features.shape))
print("矩陣", tuple(embedding.weight.shape))


# Owner supplied proportional check
other_embedding = nn.Linear(48,16)
print('out16',tuple(other_embedding(patches).shape),tuple(other_embedding.weight.shape))
print('two input collision',(torch.tensor([[1.,0.],[0.,1.]]) @ torch.tensor([0.5,0.5])).tolist())

