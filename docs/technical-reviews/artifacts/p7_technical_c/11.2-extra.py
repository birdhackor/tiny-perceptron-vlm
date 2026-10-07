original_ids={id(p) for g in optimizer.param_groups for p in g['params']}
print('same optimizer objects',original_ids=={id(p) for p in parameters_to_update})
model.language.blocks[-1].requires_grad_(True)
listed=[(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
print('unfreeze names',listed,'total',sum(n for _,n in listed),'old optimizer',sum(p.numel() for g in optimizer.param_groups for p in g['params']))
for g in [None,torch.tensor(0.)]:
 p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=.1,weight_decay=.1);p.grad=g;opt.step();print('None-vs-zero',g,p.item())
