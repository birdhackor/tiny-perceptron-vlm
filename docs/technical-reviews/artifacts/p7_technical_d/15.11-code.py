import time,torch
from tiny_perceptron.modern import DenseFFN,MoEFFN
torch.manual_seed(0);torch.set_num_threads(1);x=torch.randn(2,8,8);layers=[('Dense',DenseFFN(8,hidden=16)),('MoE',MoEFFN(8,experts=4,top_k=2,hidden=4))]
with torch.no_grad():
 for name,layer in layers:
  layer.eval()
  for _ in range(5):layer(x)
  start=time.perf_counter()
  for _ in range(30):layer(x)
  elapsed=(time.perf_counter()-start)/30;params=sum(p.numel() for p in layer.parameters());print(name,'參數',params,'毫秒/次',round(elapsed*1000,4))
