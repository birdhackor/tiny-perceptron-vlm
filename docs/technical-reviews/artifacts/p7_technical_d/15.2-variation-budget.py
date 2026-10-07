import torch
from torch import nn

experts = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(3)])
tables = [torch.eye(2), 2 * torch.eye(2), torch.tensor([[0.0, 1.0], [1.0, 0.0]])]
with torch.no_grad():
    for expert, table in zip(experts, tables):
        expert.weight.copy_(table)
x = torch.tensor([[1.0, 2.0]])
print("各自輸出", [e(x).tolist() for e in experts])
print("同一權重", experts[0].weight is experts[1].weight)
print("權重總數", sum(p.numel() for p in experts.parameters()))
shared = nn.Linear(2, 2, bias=False)
repeated = nn.ModuleList([shared, shared, shared])
print("重複引用總數", sum(p.numel() for p in repeated.parameters()))

with torch.no_grad(): experts[0].weight.mul_(3)
print('variation', [e(x).tolist() for e in experts])
import json
from tiny_perceptron.model import TinyLM,ModelConfig
r=json.load(open('docs/course-experiments/results/moe.json'))
v=r['results']['variants']['top2_aux0.01'];m=TinyLM(ModelConfig(**v['model']['config']))
print('config',v['model']['config'])
print('raw_budget',v['budget'])
print('experts',sum(p.numel() for b in m.blocks for p in b.ffn.experts.parameters()),'expert_count',sum(len(b.ffn.experts) for b in m.blocks),'all',sum(p.numel() for p in m.parameters()))
