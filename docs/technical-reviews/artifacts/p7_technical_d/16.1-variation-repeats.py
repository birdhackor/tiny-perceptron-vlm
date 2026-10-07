import torch
from torch.profiler import profile,ProfilerActivity
from tiny_perceptron.model import TinyLM,ModelConfig
torch.set_num_threads(1);model=TinyLM(ModelConfig(width=8)).eval()
print('params',sum(v.numel() for v in model.parameters()),'bytes',sum(v.numel()*v.element_size() for v in model.parameters()))
with torch.no_grad():
 for repetition in range(3):
  for length in [3,12]:
   ids=torch.arange(1,length+1)[None]
   for _ in range(3):model(ids)
   with profile(activities=[ProfilerActivity.CPU]) as p:
    for _ in range(5):result=model(ids)
   print('repeat',repetition,'length',length,'shape',tuple(result['logits'].shape))
   for e in p.key_averages():
    if e.key in ['aten::mm','aten::addmm','aten::matmul']:print(e.key,'self_us',e.self_cpu_time_total,'total_us',e.cpu_time_total,'calls',e.count,'avg_us',e.cpu_time)
