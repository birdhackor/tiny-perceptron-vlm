import torch,json
w=torch.tensor([.7,.2]); o=torch.tensor([[1.,0.],[0.,2.]])
print('swap_only_outputs',((w/w.sum())@o.flip(0)).tolist())
print('swap_both',((w.flip(0)/w.sum())@o.flip(0)).tolist())
r=json.load(open('docs/course-experiments/results/moe.json'))['results']['variants']
for n in ['top1_aux0.01','top2_aux0.01']:print(n,'training',r[n]['training']['steps'])
