import torch
from torch.profiler import profile,ProfilerActivity
from tiny_perceptron.model import TinyLM,ModelConfig
torch.set_num_threads(1);model=TinyLM(ModelConfig(width=8)).eval();ids=torch.tensor([[1,2,3]])
with torch.no_grad():
 for _ in range(3):model(ids)
 with profile(activities=[ProfilerActivity.CPU]) as p:
  for _ in range(5):result=model(ids)
print('分數形狀',tuple(result['logits'].shape));print(p.key_averages().table(sort_by='self_cpu_time_total',row_limit=5));print('權重bytes',sum(v.numel()*v.element_size() for v in model.parameters()))
