import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from torch import nn
from tiny_perceptron.multimodal import expand_modalities

embedding = nn.Embedding(264, 8)
ids = torch.tensor([1, 5, 4, 73, 2])
labels = torch.tensor([-100, -100, -100, 73, 2])
features = {5: torch.randn(3, 8)}
x, y = expand_modalities(ids, labels, embedding, features, {5})
print("輸入特徵", tuple(x.shape))
print("下一格目標", y.tolist())


# Owner supplied proportional check
more={5:torch.randn(5,8)}
xx,yy=expand_modalities(ids,labels,embedding,more,{5})
print('five image features',tuple(xx.shape),yy.tolist(),'ignored count',int((yy==-100).sum()))
print('A byte plus offset',ord('A'),ord('A')+8)

