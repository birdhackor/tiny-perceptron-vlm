import torch
from tiny_perceptron.quantization import quantize_symmetric
x=torch.tensor([0.1,0.2,0.3,2.0])
for clip in (False,True):
 s=x.clamp(-1,1) if clip else x
 q,scale=quantize_symmetric(s,bits=4)
 print('clip',clip,'scale',scale.item(),'q',q.tolist(),'restored',(q.float()*scale).tolist(),'outlier_error',(x[-1]-q[-1].float()*scale).abs().item())
