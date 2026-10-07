import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
import torch
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

model = MultiModalLM(TinyLM(ModelConfig(width=8)))
model.requires_grad_(False)
model.image_projector.requires_grad_(True)
trainable = []
parameters_to_update = []
for name, p in model.named_parameters():
    if p.requires_grad:
        trainable.append((name, p.numel()))
        parameters_to_update.append(p)
optimizer = torch.optim.AdamW(parameters_to_update, lr=0.001)
print("可訓練", trainable)
print("更新參數總數", sum(p.numel() for group in optimizer.param_groups for p in group["params"]))


# Owner supplied proportional check
original_ids={id(p) for g in optimizer.param_groups for p in g['params']}
print('same optimizer objects',original_ids=={id(p) for p in parameters_to_update})
model.language.blocks[-1].requires_grad_(True)
listed=[(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
print('unfreeze names',listed,'total',sum(n for _,n in listed),'old optimizer',sum(p.numel() for g in optimizer.param_groups for p in g['params']))
for g in [None,torch.tensor(0.)]:
 p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=.1,weight_decay=.1);p.grad=g;opt.step();print('None-vs-zero',g,p.item())

