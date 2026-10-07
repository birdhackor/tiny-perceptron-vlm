import torch
for c in [1.,2.]:
 print('equal',c,(torch.tensor([0.,0.])*c).softmax(0).tolist())
for g in [0.,.5,1.]:
 print('gamma',g,'new_angle',(1-g)*60/2+g*60)
