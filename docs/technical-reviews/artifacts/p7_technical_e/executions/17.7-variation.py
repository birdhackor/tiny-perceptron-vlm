import torch
from torch import nn
from tiny_perceptron.quantization import QuantizedLinear
l=nn.Linear(2,2,bias=False)
with torch.no_grad():l.weight.copy_(torch.tensor([[.7,.2],[-.7,1.2]]))
for b in [4,8]:
 q=QuantizedLinear(l,bits=b)
 print('bits',b,'buffers',[(n,str(t.dtype),t.numel()*t.element_size(),t.requires_grad) for n,t in q.named_buffers()],'parameters',len(list(q.parameters())),'output_dtype',str(q(torch.ones(1,2)).dtype),'storage_bytes',q.storage_bytes())
 assert len(list(q.parameters()))==0
 assert q.storage_bytes()==(10 if b==4 else 12)
