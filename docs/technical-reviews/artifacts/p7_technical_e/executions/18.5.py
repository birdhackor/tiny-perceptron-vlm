import torch
logits=torch.tensor([4.,1.,0.])
for temperature in (1.,2.,4.):
 probability=(logits/temperature).softmax(0);print('T',temperature,'機率',probability.round(decimals=4).tolist(),'第一比第二',round((probability[0]/probability[1]).item(),4))
