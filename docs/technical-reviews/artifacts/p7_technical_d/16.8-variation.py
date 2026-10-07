import torch
import torch.nn.functional as F
print('torch',torch.__version__)
for seed,invert in [(0,True),(1,False)]:
 torch.manual_seed(seed);q,k,v=[torch.randn(1,1,4,3) for _ in range(3)];mask=torch.ones(4,4,dtype=torch.bool).tril()[None,None]
 a=(q@k.transpose(-2,-1)/(3**.5)).masked_fill(~mask,float('-inf')).softmax(-1)@v
 b=F.scaled_dot_product_attention(q,k,v,attn_mask=(~mask if invert else mask),dropout_p=0.)
 close=torch.allclose(a,b,atol=1e-6);print('seed',seed,'invert',invert,'shape',tuple(b.shape),'max_error',(a-b).abs().max().item(),'assert_close',close)
 assert close != invert
 if invert:
  try:assert torch.allclose(a,b,atol=1e-6)
  except AssertionError:print('expected actual AssertionError from inverted mask')
