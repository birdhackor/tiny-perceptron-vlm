import torch
for x in [[.8,.15,.05],[.8,.05,.15]]:
 p=torch.tensor(x);print('first', ['貓','狗','鳥'][p.argmax().item()],'runner-up',['貓','狗','鳥'][p.argsort(descending=True)[1].item()],'sum',p.sum().item())
