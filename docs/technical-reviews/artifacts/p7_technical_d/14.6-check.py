import json,torch
from torch import nn
from tiny_perceptron.model import TinyLM,ModelConfig
embedding=nn.Embedding(3,2)
with torch.no_grad():embedding.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[1.,1.]]))
tied=nn.Linear(2,3,bias=False);tied.weight=embedding.weight;copied=nn.Linear(2,3,bias=False)
with torch.no_grad():copied.weight.copy_(embedding.weight);embedding.weight[1,1]+=1
h=torch.tensor([[2.,1.]]);print('variation',tied(h).tolist(),copied(h).tolist())
r=json.load(open('docs/course-experiments/results/modern.json'))['results']['variants']
for name in ['baseline','tied']:
 v=r[name];m=TinyLM(ModelConfig(**v['model']['config']));print(name,'parameters',sum(p.numel() for p in m.parameters()),'tiedis',m.output.weight is m.embedding.weight,'steps',v['training']['steps'])
 for split in ['validation','test']:
  z=v['heldout'][split];print(name,split,z['nll_sum']/z['effective_tokens'],z['effective_tokens']);assert z['nll_sum']/z['effective_tokens']==z['nll']
print('difference',141568-124672,264*64);print('torch',torch.__version__)
